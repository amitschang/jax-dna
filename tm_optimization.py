"""An example of running a melting temperature simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.oxdna.oxDNA``
"""
import functools
import logging
import shutil
import tempfile
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as jdna1_energy
import jax_dna.input.topology as jdna_top
import jax_dna.observables as jd_obs
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.optimization.simulator as jdna_simulator
import jax_dna.simulators.oxdna as oxdna
import jax_dna.utils.types as jdna_types
import jax_md
import matplotlib.pyplot as plt
import numpy as onp
import optax
import pandas as pd
from jax_dna.input import oxdna_input
from jax_dna.observables.melting_temp import MeltingTemp
from jax_dna.ui.loggers.aim import AimLogger
from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.multilogger import MultiLogger
from jax_dna.utils.units import get_kt, get_kt_from_C

jax.config.update("jax_enable_x64", True)

# Logging configurations =======================================================
logging.basicConfig(level=logging.INFO, filename="opt.log", filemode="w")
objective_logging_config = {
    "level":logging.INFO,
    "filename":"objective.log",
    "filemode":"w",
}
simulator_logging_config = objective_logging_config | {"filename": "simulator.log"}

def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.WARNING)

    # Sim configuration and Energy Function ==========================================================

    input_dir = Path("data/templates/tm-8bp-2op")

    umbrella_config = {
        "n_steps": 5_000,
        "batch_size": 1,
        "order_parameter_shape": (10, 2),
        "extrap_temp_range": get_kt(jnp.linspace(280, 350, 10))
    }
    kt_range = umbrella_config["extrap_temp_range"]
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    sim_config = oxdna_input.read(input_dir / "input")  # just to check it exists
    kT_C = float(sim_config["T"].replace("C", ""))
    kT = get_kt_from_C(kT_C)
    batch_size = umbrella_config["batch_size"]
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
    # ==========================================================================


    # Simulators ================================================================


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

    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=transform_fn,
        sim_temperature=kT, #this should be in simulation units
        temperature_range=kt_range,
        energy_config=energy_configs,
        energy_fn_builder=obj_energy_fn_builder,
        topology=top #range of temperatures to extrapolate simulation to via histogram reweighting
    )

    class EnergyInfo(pd.DataFrame):
        pass

    def melting_temp_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
        opt_params: jdna_types.Params,
        extra: dict[str, typing.Any] = None, # observable map
    ) -> tuple[float, tuple[str, typing.Any]]:
        infos = [i for i in extra.values() if isinstance(i, EnergyInfo)][0]
        umbrella_weights = infos["op_weight"].values
        op_values = infos[["op1", "op2"]].values

        obs = melting_temp_fn(traj, umbrella_weights, (9, 2), op_values, opt_params)

        expected_melting_temp = jnp.dot(weights, obs).sum()
        loss = (expected_melting_temp - jd_obs.melting_temp.TARGETS["SL_avg_8bp"]) ** 2
        loss = jnp.sqrt(loss)
        return loss, (("melting_temp", expected_melting_temp), {})


    melting_temp_objective = jdna_objective.DiffTReObjective(
        name="prop_twist",
        required_observables=['traj-sim0', 'energy-sim0'],
        needed_observables=['traj-sim0', 'energy-sim0'],
        logging_observables=["loss", "melting_temp", "neff"],
        grad_or_loss_fn=melting_temp_loss_fn,
        energy_fn_builder=obj_energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
    )

    def simdir_from_inputs(path):
        output_dir = tempfile.mkdtemp()
        shutil.copytree(path, output_dir, dirs_exist_ok=True)
        return Path(output_dir)

    oxdna_simulator = oxdna.oxDNASimulator(
        input_dir=simdir_from_inputs(input_dir),
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_configs=energy_configs,
        source_path="../oxDNA"
    )

    def sim_fun(params, meta_data):
        traj = oxdna_simulator.run(params, seed=2104045939) #or seed=841595951) # This is a seed that produces both bound and unbound states
        energy_df_columns = [
            "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot",
            "acc_ratio_vol", "op1", "op2", "op_weight"
        ]
        energy_df = EnergyInfo(
            pd.read_csv(oxdna_simulator.input_dir / "energy.dat", names=energy_df_columns, sep='\s+', skiprows=1)
        )
        return traj, energy_df

    simulator = jdna_simulator.BaseSimulator(
        name="oxdna-sim",
        fn=sim_fun,
        exposes = ['traj-sim0', 'energy-sim0'],
        meta_data = {},
    )

    optimizer = jdna_optimization.SimpleOptimizer(
        objective=melting_temp_objective,
        simulator=simulator,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    aim_logger = AimLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])
    aim_logger.aim_run.set("learning_rate", 1e-4)

    for i in range(100):
        state, opt_params, grad = optimizer.step(opt_params)

        for metric, value in optimizer.objective.logging_observables():
            logger.log_metric(metric, value, i)

        optimizer.post_step(state, opt_params)


    f_infs = melting_temp_fn.get_melting_curve(combined_trajectory,
        umbrella_weights=umbrella_weights,
        order_parameter_shape=umbrella_config["order_parameter_shape"],
        bin_indices=op_values,
        energy_fn_builder_fn=energy_fn_builder_fn,
        opt_params=opt_params)


    print("testing melting_temp:", f_infs)
    plt.plot(f_infs[0], f_infs[1])
    plt.show()
    # amalgamate histograms
    histogram_data, hist_dict = collect_histogram_data(sim_outputs_dir, batch_size)
    plt.bar(onp.arange(len(histogram_data.T[2])),histogram_data.T[2])
    plt.show()

    weight_multiplier = {}
    norm = sum(hist_dict.values())
    #compute new weights
    for key in hist_dict:
        weight_multiplier[key] = jnp.where(hist_dict[key]==0,
                       2.,
                       norm/hist_dict[key])

    new_weights = {}
    #read previous weights
    old_weights = onp.loadtxt(input_wfile_fname,dtype=[('col1', int), ('col2', int), ('col3', float)])
    for row in old_weights:
        key = (int(row[0]), int(row[1]))
        value = float(row[2])
        new_weights[key] = value*weight_multiplier[key]

    def normalize_dict(weights_dict):
        min_value = min(weights_dict.values())
        normalized_weights = {key: value / min_value for key, value in weights_dict.items()}
        return normalized_weights
    normalized_weights = normalize_dict(new_weights)

    #re-write wfile.txt with new weights, saving the previous weights to a new file
    shutil.move(input_dir / "wfile.txt", input_dir / "wfile_0.txt")
    with open(input_dir / "wfile.txt", 'a') as f:
         for key in new_weights:
               i, j = key
               value = float(normalized_weights[key])
               f.write(f"{i} {j}  {value:.6g}\n")


    #check for convergence of weights


if __name__ == "__main__":
    main()




