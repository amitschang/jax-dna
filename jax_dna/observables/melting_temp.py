"""Melting temperature observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax import lax

import jax_dna.input.topology as jdna_top
import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types
from jax_dna.energy import configuration
from jax_dna.utils.units import get_kt_from_C

TARGETS = {
    "SL_avg_6bp": get_kt_from_C(31.2),  # degrees
    "SL_avg_8bp": get_kt_from_C(48.2),  # degrees
    "SL_avg_12bp": get_kt_from_C(64.7),  # degrees
}
def jax_interp1d(x, y, x_new):
    """Simple linear interpolation function using JAX.

    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        x_new: Point(s) at which to interpolate

    Returns:
        Interpolated y value(s)
    """
    # Sort x and y if x is not already sorted
    sorted_idx = jnp.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    # Find indices of lower points
    idx = jnp.searchsorted(x_sorted, x_new) - 1
    # Clip to ensure indices are within bounds
    idx = jnp.clip(idx, 0, len(x_sorted) - 2)

    # Get lower and upper points
    x_low = x_sorted[idx]
    x_high = x_sorted[idx + 1]
    y_low = y_sorted[idx]
    y_high = y_sorted[idx + 1]

    # Linear interpolation
    t = (x_new - x_low) / (x_high - x_low)
    return y_low + t * (y_high - y_low)

def compute_finf(ratio):
    finf = 1 + 1/(2*ratio) - jnp.sqrt((1 + 1/(2*ratio))**2 - 1)
    return finf

def find_melting_temp(temperatures, ratios, target_ratio=0.5):
    """Find the temperature at which the concentration of single strands = 0.5 * duplex concentration
    Args:
        temperatures: Array of temperature values
        ratios: Array of unbound:bound ratio values corresponding to each temperature
        target_ratio: The ratio value to find (default: 0.5)

    Returns:
        The interpolated temperature where ratio = target_ratio
    """
    target_temperature = jax_interp1d(ratios, temperatures, 0.5)
    return target_temperature

def compute_curve_width(temperatures, ratios):
    """Find the width of the melting curve, defined as the temperature separation between unbound:bound ratio = 0.2
    and unbound:bound ratio = 0.8
    Args:
        temperatures: Array of temperature values
        ratios: Array of unbound:bound ratio values corresponding to each temperature
    Returns:
        The width of the interpolated temperature curve between 0.2 and 0.8
    """
    width = jax_interp1d(ratios, temperatures, 0.7) - jax_interp1d(ratios, temperatures, 0.3)
    return width

# has access to rigid_body_transform_fn
@chex.dataclass(frozen=True)
class MeltingTemp(jd_obs.BaseObservable):
    """Computes the melting temperature of a duplex from trajectory data and umbrella sampling weights.

    The melting temperature is defined as the temperature at which the concentration
    of DNA duplexes is double that of the concentration of single strands.

    Args:
    - sim_temperature: float. the temperature at which the SimulatorTrajectory was collected, in sim. units
    - temperature range: a vector containing the temperatures to extrapolate the SimulatorTrajectory data to (via histogram reweighting) in Kelvin
    """

    sim_temperature: float  # Temperature at which the simulation was conducted in sim. units
    temperature_range: jnp.ndarray = dc.field(hash=False)
    energy_config: list[configuration.BaseConfiguration] #we need this so we know where to replace Ts during temp extrapolation
    topology: jdna_top.Topology #topology of the relevant system
    energy_fn_builder: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]]
    params_dict_list: list = dc.field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)
        for ec in self.energy_config:
            current_dict = ec.to_dictionary(include_dependent=True, exclude_non_optimizable=False)
            # Check if "kt" exists in the dictionary to find any temp-dependent energy terms
            if "kt" in current_dict:
                # Append the placeholder dictionary {"kt": 0} to params_dict_list
                self.params_dict_list.append({"kt": 0})
            else:
                self.params_dict_list.append({})

    def __call__(
        self,
        trajectory: jd_sio.SimulatorTrajectory, #!
        umbrella_weights: jnp.ndarray,
        order_parameter_shape: jnp.ndarray,
        bin_indices: jnp.ndarray,
        #energy_fn_builder_fn: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params: jd_types.PyTree, #!
    ) -> float:
        """Calculate the melting temperature.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the melting temperature for
            umbrella_weights (jnp.ndarray): an N-dimensional array containing umbrella sampling weights
            bin_indices (jnp.ndarray): an array of length (num_timesteps x number of umbrella order params)
                                        containing the umbrella sampling bins for each configuration in the SimulatorTrajectory
            energy_fn_builder_fn: the function that builds a energy function at sim_temperature
            opt_params: the parameters to optimize; use the current vals in building the energy functions

        Returns:
            float: the melting temperature in Kelvins
        """
        """Calculate the melting temperature (default observable output)."""
        return self.get_melting_temperature(trajectory, umbrella_weights, order_parameter_shape, bin_indices, self.energy_fn_builder, opt_params)

    def get_extrap_ratios(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        umbrella_weights: jnp.ndarray,
        order_parameter_shape: jnp.ndarray, #e.g. shape (2, n_bps) for 2D umbrella sampling with mindistance and # bps
        bin_indices: jnp.ndarray, #an array of tuples
        energy_fn_builder: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the bound:unbound ratios at the extrapolated temperatures."""
        # convert all Kelvin to oxDNA sim units
        kT_range = self.temperature_range
        sim_kT = self.sim_temperature

        #compute energy functions at T0 and T

        # def energy_fn_builder(params: jd_types.Params) -> callable:
        #     base_energy_fn = energy_fn_builder_fn(params)
        #     def process_single_frame(frame):
        #         return base_energy_fn(
        #             frame.rigid_body,
        #             seq=jnp.array(self.topology.seq),
        #             bonded_neighbors=self.topology.bonded_neighbors,
        #             unbonded_neighbors=self.topology.unbonded_neighbors.T
        #             ) #/ self.topology.n_nucleotides #FIXME: check this?
        #     return jax.vmap(process_single_frame)


        energy_at_T0 = energy_fn_builder(opt_params)
        energies_T0 = energy_at_T0(trajectory)

        #find the unbiased ratio of bound:unbound
        def finf_at_T(extrapolated_temp: float):
            """Calculate the ratio bound:unbound at temperature T"""
            extrap_T_params = []
            for val in self.params_dict_list:
                if "kt" in val:
                    new_val = dict(val)  # Create a copy
                    new_val["kt"] = extrapolated_temp
                    extrap_T_params.append(new_val)
                else:
                    extrap_T_params.append(val)

            merged_params = []
            for dict1, dict2 in zip(opt_params, extrap_T_params, strict=False):
                # Merge dictionaries - dict2 values take precedence for duplicate keys
                merged_dict = dict1 | dict2
                merged_params.append(merged_dict)


            energy_T_extrap = energy_fn_builder(merged_params) #this merges the dictionaries, adding the kT we want to extrapolate to. #FIXME: does this mess up the gradient tracing? Or is kt still non-optimizable?
            energies_T_extrap = energy_T_extrap(trajectory) #might need to be trajectory.rigid_body

            boltz_factor = jnp.exp((energies_T0/sim_kT) - (energies_T_extrap/extrapolated_temp))
            unbiased_counts = (1 / umbrella_weights) * boltz_factor
            total_unbound = jnp.where(bin_indices[:, 0] == 0, unbiased_counts, 0).sum()
            total_bound = jnp.where(bin_indices[:, 0] != 0, unbiased_counts, 0).sum()
            phi = total_bound / total_unbound
            f_inf = compute_finf(phi) # apply finite size correction

            return f_inf

        #extrapolate the histogram (bound:unbound) to temperature_range:

        extrap_ratios = jax.vmap(finf_at_T)(kT_range)
        return extrap_ratios


    def get_melting_temperature(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        umbrella_weights: jnp.ndarray,
        order_parameter_shape: jnp.ndarray, #e.g. shape (2, n_bps) for 2D umbrella sampling with mindistance and # bps
        bin_indices: jnp.ndarray, #an array of tuples
        energy_fn_builder_fn: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the melting temperature."""
        kT_range = self.temperature_range #-> returns Boltzmann's constant * temp in oxDNA units
        extrap_ratios = self.get_extrap_ratios(trajectory, umbrella_weights, order_parameter_shape, bin_indices, energy_fn_builder_fn, opt_params)
        #interpolate the melting curve
        melting_T = find_melting_temp(kT_range, extrap_ratios)
        return melting_T

    def get_melting_curve(
       self,
       trajectory: jd_sio.SimulatorTrajectory,
       umbrella_weights: jnp.ndarray,
       order_parameter_shape: jnp.ndarray, #e.g. shape (2, n_bps) for 2D umbrella sampling with mindistance and # bps
       bin_indices: jnp.ndarray, #an array of tuples
       energy_fn_builder_fn: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
       opt_params: jd_types.PyTree,
   ) -> float:
       """Calculate the melting curve."""
       extrap_ratios = self.get_extrap_ratios(trajectory, umbrella_weights, order_parameter_shape, bin_indices, energy_fn_builder_fn, opt_params)
       return self.temperature_range, extrap_ratios

    def get_melting_curve_width(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        umbrella_weights: jnp.ndarray,
        order_parameter_shape: jnp.ndarray, #e.g. shape (2, n_bps) for 2D umbrella sampling with mindistance and # bps
        bin_indices: jnp.ndarray, #an array of tuples
        energy_fn_builder_fn: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the melting curve width."""
        extrap_ratios = self.get_extrap_ratios(trajectory, umbrella_weights, order_parameter_shape, bin_indices, energy_fn_builder_fn, opt_params)
        kT_range = self.temperature_range #-> returns Boltzmann's constant * temp in oxDNA units
        #interpolate the melting curve
        melting_curve_width = compute_curve_width(kT_range, extrap_ratios)
        return melting_curve_width
