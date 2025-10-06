from pathlib import Path
import subprocess

import numpy as np
from jax_dna.input.topology import from_oxdna_file
from jax_dna.input.trajectory import from_file
from jax_dna.simulators.base import BaseSimulation
from jax_dna.simulators.io import SimulatorTrajectory


replacement_map = {
    "bond_coeff *": ("eps_backbone", "delta_backbone", "r0_backbone"),
    "pair_coeff * * oxdna/excv": (
        "eps_exc", "sigma_backbone", "dr_star_backbone",
        "eps_exc", "sigma_back_base", "dr_star_back_base",
        "eps_exc", "sigma_base", "dr_star_base"
    ),
    "pair_coeff * * oxdna/stk": (
        None, None, "eps_stack_base", "eps_stack_kt_coeff", "a_stack",
        "dr0_stack", "dr_c_stack", "dr_low_stack", "dr_high_stack",
        "a_stack_4", "theta0_stack_4", "delta_theta_star_stack_4",
        "a_stack_5", "theta0_stack_5", "delta_theta_star_stack_5",
        "a_stack_6", "theta0_stack_6", "delta_theta_star_stack_6",
        "a_stack_1", "neg_cos_phi1_star_stack",
        "a_stack_2", "neg_cos_phi2_star_stack"
    ),
    "pair_coeff * * oxdna/hbond": (
        None, "HYDR_F1", "a_hb", "dr0_hb", "dr_c_hb", "dr_low_hb", "dr_high_hb",
        "a_hb_1", "theta0_hb_1", "delta_theta_star_hb_1",
        "a_hb_2", "theta0_hb_2", "delta_theta_star_hb_2",
        "a_hb_3", "theta0_hb_3", "delta_theta_star_hb_3",
        "a_hb_4", "theta0_hb_4", "delta_theta_star_hb_4",
        "a_hb_7", "theta0_hb_7", "delta_theta_star_hb_7",
        "a_hb_8", "theta0_hb_8", "delta_theta_star_hb_8"
    ),
    "pair_coeff 1 4 oxdna/hbond": (
        None, "eps_hb", "a_hb", "dr0_hb", "dr_c_hb", "dr_low_hb", "dr_high_hb",
        "a_hb_1", "theta0_hb_1", "delta_theta_star_hb_1",
        "a_hb_2", "theta0_hb_2", "delta_theta_star_hb_2",
        "a_hb_3", "theta0_hb_3", "delta_theta_star_hb_3",
        "a_hb_4", "theta0_hb_4", "delta_theta_star_hb_4",
        "a_hb_7", "theta0_hb_7", "delta_theta_star_hb_7",
        "a_hb_8", "theta0_hb_8", "delta_theta_star_hb_8"
    ),
    "pair_coeff 2 3 oxdna/hbond": (
        None, "eps_hb", "a_hb", "dr0_hb", "dr_c_hb", "dr_low_hb", "dr_high_hb",
        "a_hb_1", "theta0_hb_1", "delta_theta_star_hb_1",
        "a_hb_2", "theta0_hb_2", "delta_theta_star_hb_2",
        "a_hb_3", "theta0_hb_3", "delta_theta_star_hb_3",
        "a_hb_4", "theta0_hb_4", "delta_theta_star_hb_4",
        "a_hb_7", "theta0_hb_7", "delta_theta_star_hb_7",
        "a_hb_8", "theta0_hb_8", "delta_theta_star_hb_8"
    ),
    "pair_coeff * * oxdna/xstk": (
        "k_cross", "r0_cross", "dr_c_cross", "dr_low_cross", "dr_high_cross",
        "a_cross_1", "theta0_cross_1", "delta_theta_star_cross_1",
        "a_cross_2", "theta0_cross_2", "delta_theta_star_cross_2",
        "a_cross_3", "theta0_cross_3", "delta_theta_star_cross_3",
        "a_cross_4", "theta0_cross_4", "delta_theta_star_cross_4",
        "a_cross_7", "theta0_cross_7", "delta_theta_star_cross_7",
        "a_cross_8", "theta0_cross_8", "delta_theta_star_cross_8"
    ),
    "pair_coeff * * oxdna/coaxstk": (
        "k_coax", "dr0_coax", "dr_c_coax", "dr_low_coax", "dr_high_coax",
        "a_coax_1", "theta0_coax_1", "delta_theta_star_coax_1",
        "a_coax_4", "theta0_coax_4", "delta_theta_star_coax_4",
        "a_coax_5", "theta0_coax_5", "delta_theta_star_coax_5",
        "a_coax_6", "theta0_coax_6", "delta_theta_star_coax_6",
        "a_coax_3p", "cos_phi3_star_coax",
        "a_coax_4p", "cos_phi4_star_coax"
    ),
}

class LAMMPSOxDNA(BaseSimulation):

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.input_lines = self.input_dir.joinpath("input").read_text().splitlines()

    def run(self, params: dict[str, float], seed: int = None) -> Path:
        self._replace_parameters(params, seed)
        subprocess.check_call(["lmp", "-in", "input"], cwd=self.input_dir)

        # read traj - look at tacoxdna LAMMPS_oxDNA - unfortunately this package
        # is not installable, and is designed as executable only.
        # A hack for now
        data_file = next(self.input_dir.glob("data.*")).resolve()
        out_file = next(self.input_dir.glob("out.*")).resolve()
        subprocess.check_call(
            ["python", "LAMMPS_oxDNA.py", str(data_file), str(out_file)],
            cwd="/Users/arik/ws/ssec/tacoxdna/src")
        top_file = Path("/Users/arik/ws/ssec/tacoxdna/src") / f"{data_file.name}.top"
        traj_file = Path("/Users/arik/ws/ssec/tacoxdna/src") / f"{data_file.name}.oxdna"
        topology = from_oxdna_file(top_file)
        traj = from_file(traj_file, topology.strand_counts, is_oxdna=False)
        # end hack
        return SimulatorTrajectory(
            rigid_body = traj.state_rigid_body,
        )

    def _replace_parameters(self, params, seed):
        flat_params = {k: v for i in params for k,v in i.items()}
        new_lines = []
        for line in self.input_lines:
            if line.split()[:3] == ["variable", "seed", "equal"]:
                if seed is not None:
                    line = f"variable seed equal {seed}"
                else:
                    seed = np.random.randint(0, 2**24)
                    line = f"variable seed equal {seed}"
            for key, replacements in replacement_map.items():
                if line.startswith(key):
                    parts = line.removeprefix(key).split()
                    new_parts = []
                    for part, replacement_param in zip(parts, replacements, strict=True):
                        if replacement_param is None or replacement_param not in flat_params:
                            new_parts.append(part)
                        else:
                            new_parts.append(f"{flat_params[replacement_param]:f}")
                    line = f"{key} {' '.join(new_parts)}"
            new_lines.append(line)
        self.input_dir.joinpath("input").write_text("\n".join(new_lines))

