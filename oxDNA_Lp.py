"""An example of running a simple optimization using oxDNA and the DiffTRE algorithm.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.advanced_optimizations.oxDNA.oxDNA``
"""


"""Each calculation of $Lps$ involves 64 parallel simulations each comprising $2.5 \times 10^6$ timesteps, sampling reference states every $10^4$ steps.
We neglect the first 2 base pairs on either end of the duplex due to boundary effects and only fit Equation \ref{eqn:lps-exp} for $m \leq 40$ as large $m$ have poor statistics.
We again use a learning rate of $0.001$ and resample states via the same protocol as described for pitch optimizations.
"""

import functools
import logging
from pathlib import Path
import shutil
import tempfile
import typing
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray
from tqdm import tqdm
import operator

import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.topology as jdna_top
import jax_dna.observables as jd_obs
import jax_dna.observables.persistence_length as persistence_length
import jax_dna.observables.base as base
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.simulators.oxdna as oxdna
from jax_dna.ui.loggers.aim import AimLogger
from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.multilogger import MultiLogger
import jax_dna.utils.types as jdna_types

jax.config.update("jax_enable_x64", True)


# Logging configurations =======================================================
logging.basicConfig(level=logging.INFO, filename="opt.log", filemode="w")
objective_logging_config = {
    "filename":"objective.log",
    "filemode":"w",
}
simulator_logging_config = objective_logging_config | {"filename": "simulator.log"}
# ==============================================================================


# To combine the gradients of multiple objectives, we can use a mean, however
# this example only has one objective, so it will remain unchanged.
def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
    if len(trees) <= 1:
        return trees[0]
    summed = jax.tree.map(operator.add, *trees)
    return jax.tree.map(lambda x: x / len(trees), summed)

def main():
    # The coordination of objectives and simulators is done through Ray actors.
    # So we need to initialize a ray server
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",
                "JAX_PLATFORM_NAME": "cpu",
            }
        }
    )

    # Input configuration ======================================================
    input_dir = Path("data/templates/simple-helix-60bp-oxdna1")
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")

    optimization_config = {
        "n_steps": 1_00,
        "n_opt_steps": 25,
        "oxdna_build_threads": 4,
        "log_every": 10,
        "n_oxdna_runs": 10,
    }

    TARGET_LP = 40. #40 nm
    simulation_config, energy_config = dna1_energy.default_configs()
    kT = simulation_config["kT"]

    # Energy Function ==========================================================
    energy_fns = dna1_energy.default_energy_fns()
    energy_fn_configs = []
    opt_params = []
    for ec in dna1_energy.default_energy_configs():
        # We are only interested in the stacking configuration
        # However we don't want to optimize ss_stack_weights and kt
        if isinstance(ec, dna1_energy.StackingConfiguration):
            ec = ec.replace(
                non_optimizable_required_params=(
                    "ss_stack_weights",
                    "kt",
                )
            )
            opt_params.append(ec.opt_params)
            energy_fn_configs.append(ec)
        elif isinstance(ec, dna1_energy.HydrogenBondingConfiguration):
            ec = ec.replace(
                non_optimizable_required_params=(
                    "ss_hb_weights",
                )
            )
            opt_params.append(ec.opt_params)
            energy_fn_configs.append(ec)
        else:
            energy_fn_configs.append(ec)
            opt_params.append({})


    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_fn_configs,
        transform_fn=transform_fn,
    )

    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    # ==========================================================================

    # Simulators ================================================================
    @ray.remote
    class RaySimulator:
        def __init__(self, **kwargs):
            self.simulator = oxdna.oxDNASimulator(**kwargs)

        def run(self, params, meta_data=None):
            return self.simulator.run(params, meta_data)

        def get_hist(self):
            hist_file = self.simulator.input_dir / self.simulator.input_config["last_hist_file"]
            hist_df_columns = ["bind", "mindist", "unbiased"]
            hist_df = pd.read_csv(hist_file, names=hist_df_columns, sep='\s+', usecols=[0,1,3], skiprows=1).set_index(["bind", "mindist"])
            hist_df["unbiased_normed"] = hist_df["unbiased"] / hist_df["unbiased"].sum()
            return hist_df

        def update_weights(self, weights):
            weights_file = self.simulator.input_dir / self.simulator.input_config["weights_file"]
            weights.to_csv(weights_file, sep=' ', header=False)

    # Make a wrapper class to run all of these remote simulators as though they
    # are a single simulator, but implement the simulator interface that has
    # exposes function so it can be used in the optimizer
    class MultiRaySimulator:
        def __init__(self, simulators):
            self.simulators = simulators

        def run(self, params, meta_data=None):
            # Run all the simulators in parallel and wait for all to be finished
            # before gathering results here
            futures = [sim.run.remote(params, meta_data) for sim in self.simulators]
            return ray.get(futures)

        def exposes(self):
            # each simulator returns 2 observables: traj and energy, but doesn't
            # really matter what they are called here, just they are unique
            return [f"obs-{i}" for i in range(len(self.simulators))]


    def simdir_from_inputs(path):
        output_dir = tempfile.mkdtemp()
        shutil.copytree(path, output_dir, dirs_exist_ok=True)
        return Path(output_dir)

    # Construct multi simulator, it expects a list of simulator actors to be
    # passed in
    multi_simulator = MultiRaySimulator([
        RaySimulator.options(num_cpus=1).remote(
            input_dir=simdir_from_inputs(input_dir),
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_configs=energy_fn_configs,
            source_path="../oxDNA"
        )
        for _ in range(optimization_config["n_oxdna_runs"])
    ])


    # Objective ================================================================
    lp_fn = jd_obs.persistence_length.PersistenceLength(
        rigid_body_transform_fn=transform_fn,
        displacement_fn = jax_md.space.free()[0],
        quartets=base.get_duplex_quartets(int(top.n_nucleotides / 2)),
        )

    def lp_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
        *args,
        **kwargs,
    ) -> tuple[float, tuple[str, typing.Any]]:
        all_corrs, all_l0s = lp_fn(traj, skip_ends=True)
        weighted_corr_mean = jnp.dot(weights, all_corrs) #DiffTRE weighting
        weighted_l0_mean = jnp.dot(weights, all_l0s)

        fit_lp, fit_offset = persistence_length.persistence_length_fit(weighted_corr_mean[:40], weighted_l0_mean) # only consider m<=40 correlations

        loss = (fit_lp - TARGET_LP) ** 2
        loss = jnp.sqrt(loss)
        return loss, (("persistence_length", fit_lp), {})

    persistence_length_objective = jdna_objective.DiffTReObjective(
        name="persistence_length",
        required_observables=multi_simulator.exposes(),
        needed_observables=multi_simulator.exposes(),
        logging_observables=["loss", "persistence_length", "neff"],
        grad_or_loss_fn=lp_loss_fn,
        energy_fn_builder=energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
        max_valid_opt_steps=10,
    )
    # ==========================================================================

    opt = jdna_optimization.SimpleOptimizer(
        objective=persistence_length_objective,
        simulator=multi_simulator,
        optimizer = optax.adam(learning_rate=1e-3),
    )
    # ==========================================================================

    aim_logger = AimLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])

    # Run optimization =========================================================
    for i in tqdm(range(optimization_config["n_opt_steps"]), desc="Optimizing"):
        opt_state, opt_params, grads = opt.step(opt_params)

        for (name, value) in opt.objective.logging_observables():
            logger.log_metric(name, value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )

if __name__=="__main__":
    main()