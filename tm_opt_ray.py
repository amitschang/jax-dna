"""An example of running a melting temperature simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

python tm_opt_ray.py
"""
import argparse
import functools
import itertools
import logging
import shutil
import tempfile
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import ray
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as jdna1_energy
import jax_dna.input.topology as jdna_top
import jax_dna.observables as jd_obs
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.simulators.oxdna as oxdna
import jax_dna.utils.types as jdna_types
import jax_md
import optax
import pandas as pd
from jax_dna.input import oxdna_input
from jax_dna.observables.melting_temp import MeltingTemp
from jax_dna.ui.loggers.aim import AimLogger
from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.multilogger import MultiLogger
from jax_dna.utils.units import get_kt, get_kt_from_string


jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.WARNING)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--num-sims",
        type=int,
        default=10,
        help="Number of parallel simulators to run.",
    )
    arg_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    arg_parser.add_argument(
        "--opt-steps",
        type=int,
        default=100,
        help="Number of optimization steps.",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = arg_parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ray.init()

    input_dir = Path("data/templates/tm-6bp-2op")

    umbrella_config = {
        "n_steps": 5_000,
        "batch_size": 1,
        "order_parameter_shape": (10, 2),
        "extrap_temp_range": get_kt(jnp.linspace(280, 350, 10))
    }
    kt_range = umbrella_config["extrap_temp_range"]
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    sim_config = oxdna_input.read(input_dir / "input")
    kT = get_kt_from_string(sim_config["T"])

    # Setup the energy functions and configs
    _, energy_config = jdna1_energy.default_configs()
    energy_fns = jdna1_energy.default_energy_fns()
    energy_configs = []
    opt_params = []

    for ec in jdna1_energy.default_energy_configs():
        if isinstance(ec, jdna1_energy.StackingConfiguration):
            ec = ec.replace( non_optimizable_required_params=(
                    "ss_stack_weights",
                    "kt",
                ),
                kt=kT)
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        elif isinstance(ec, jdna1_energy.HydrogenBondingConfiguration):
            ec = ec.replace(non_optimizable_required_params=("ss_hb_weights") )
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        else:
            energy_configs.append(ec)
            opt_params.append(ec.opt_params)

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        jdna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
    )

    # This seems like a common pattern
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    def obj_energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    # Convenience function to copy inputs to a temp directory. Each simulator
    # write to such a directory and modifies input, so having a distinct
    # location and copy for each is critical.
    #
    # Might consider having the oxdna simulator do this internally, since it
    # seems like an important pattern.
    def simdir_from_inputs(path):
        output_dir = tempfile.mkdtemp()
        shutil.copytree(path, output_dir, dirs_exist_ok=True)
        return Path(output_dir)

    # This is a convenience wrapper just to use for filtering the observables
    # for the type we desire. Observables looks generic, while some objective
    # like the difftre expects trajectories. To keep generality we probably want
    # some way to tag observables, so we can filter relevant ones.
    #
    # For this difftre has been modified to filter SimulatorTrajectory for its
    # building of the combined trajectory, enabling us to pass more observables.
    class EnergyInfo(pd.DataFrame):
        pass

    # Create a simple class for running the simulator remotely, while also
    # modifying the run function to return both trajectory and energy data, for
    # use with the melting temp objective. The chex.dataclass decorator is not
    # compatible with ray actors, so we keep the underlying simulator as an
    # internal attribute.
    @ray.remote
    class RaySimulator:
        def __init__(self, **kwargs):
            self.simulator = oxdna.oxDNASimulator(**kwargs)

        def run(self, params, meta_data=None):
            traj = self.simulator.run(params, meta_data)
            energy_df_columns = [
                "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot",
                "acc_ratio_vol", "op1", "op2", "op_weight"
            ]
            energy_df = EnergyInfo(
                pd.read_csv(self.simulator.input_dir / "energy.dat", names=energy_df_columns, sep='\s+', skiprows=1)
            )
            # note we directly pass the python objects to objective here as
            # opposed to files.
            return traj, energy_df

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
            results = ray.get(futures)
            # Flatten the list, [traj, energy, traj, energy, ...]
            observables = list(itertools.chain.from_iterable(results))
            # Prior to next run, update the umbrella weights based on the histograms
            self.update_weights()
            return observables

        def update_weights(self):
            hist = ray.get([simulator.get_hist.remote() for simulator in self.simulators])
            hist = pd.concat(hist).reset_index().groupby(["bind", "mindist"]).sum()
            weights = hist.query("unbiased_normed > 0").eval("weights = 1 / unbiased_normed")
            weights["weights"] /= weights["weights"].min()  # for numerical stability
            weights = weights[["weights"]]
            # fill in zeroed states
            weights = weights.reindex(hist.index, fill_value=0)
            # Update these in all simulators
            ray.get([simulator.update_weights.remote(weights) for simulator in self.simulators])

        def exposes(self):
            # each simulator returns 2 observables: traj and energy, but doesn't
            # really matter what they are called here, just they are unique
            return [f"obs-{i}" for i in range(2*len(self.simulators))]

    # Construct multi simulator, it expects a list of simulator actors to be
    # passed in
    multi_simulator = MultiRaySimulator([
        RaySimulator.options(num_cpus=1).remote(
            input_dir=simdir_from_inputs(input_dir),
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_configs=energy_configs,
            source_path="../oxDNA"
        )
        for _ in range(args.num_sims)
    ])

    # Setup the melting temp function, loss and objective
    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=transform_fn,
        sim_temperature=kT, #this should be in simulation units
        temperature_range=kt_range,
        energy_config=energy_configs,
        energy_fn_builder=obj_energy_fn_builder,
        topology=top #range of temperatures to extrapolate simulation to via histogram reweighting
    )

    def melting_temp_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
        opt_params: jdna_types.Params,
        observables: dict[str, typing.Any] = None, # observable map
    ) -> tuple[float, tuple[str, typing.Any]]:
        # Objective has been modified to pass in all observables as an ordered
        # dict to the loss function. Since this includes the trajectories as
        # well (and potentially other observables), we need to filter out the
        # energy data observables, we do so by looking for the EnergyInfo type.
        infos = pd.concat([i for i in observables.values() if isinstance(i, EnergyInfo)])
        umbrella_weights = infos["op_weight"].values
        op_values = infos[["op1", "op2"]].values

        # The (9, 2) for order parameter shape actually just needs to be larger
        # than the maximum order parameter in each dimension in the wfile. In
        # the future this may not need to be passed, the critical thing is we
        # look for where the bond op is zero/non-zero
        obs = melting_temp_fn(traj, umbrella_weights, (9, 2), op_values, opt_params)

        expected_melting_temp = jnp.dot(weights, obs).sum()
        loss = (expected_melting_temp - jd_obs.melting_temp.TARGETS["SL_avg_6bp"]) ** 2
        loss = jnp.sqrt(loss)
        if not jnp.isfinite(loss):
            # There is no recovery from this...
            raise ValueError("Non-finite loss encountered.")
        return loss, (("melting_temp", expected_melting_temp), {})

    melting_temp_objective = jdna_objective.DiffTReObjective(
        name="prop_twist",
        required_observables=multi_simulator.exposes(),
        needed_observables=multi_simulator.exposes(), # do we need to supply both?
        logging_observables=["loss", "melting_temp", "neff"],
        grad_or_loss_fn=melting_temp_loss_fn,
        energy_fn_builder=obj_energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
    )

    optimizer = jdna_optimization.SimpleOptimizer(
        objective=melting_temp_objective,
        simulator=multi_simulator,
        optimizer=optax.adam(learning_rate=args.learning_rate),
    )

    # Create loggers, we've added an AIM logger (https://aimstack.io/) for
    # logging to the tracking and visualization system they provide. For this,
    #  pip install aim
    # is required. We've also added a multi-logger that logs to any number of
    # loggers at once, here we use it to log to both the console and AIM.
    aim_logger = AimLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])
    # Aim has the concept of run parameters, add the learning rate to this one
    aim_logger.aim_run.set("learning_rate", args.learning_rate)

    for i in range(args.opt_steps):
        state, opt_params, _ = optimizer.step(opt_params)

        for metric, value in optimizer.objective.logging_observables():
            logger.log_metric(metric, value, i)

        optimizer.post_step(state, opt_params)


if __name__ == "__main__":
    main()




