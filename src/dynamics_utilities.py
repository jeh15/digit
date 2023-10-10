import numpy as np
import numpy.typing as npt

# Types:
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Context


def get_dynamics(
    plant: MultibodyPlant,
    context: Context,
    q: npt.ArrayLike,
    qd: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MultibodyPlant, Context]:
    """
    Computes the mass matrix, coriolis matrix, and generalized
    forces due to gravity from based on the generalized positions 
    and velocities.

    Args:
        plant: The MultibodyPlant.
        context: Context of plant.
        q: The generalized positions.
        qd: The generalized velocities.

    Returns:
        M: The mass matrix.
        C: The coriolis matrix.
        tau_g: The generalized forces due to gravity.
        plant: updated plant
        context: updated context

    """

    # Update plant model to the current state:
    plant.SetPositions(
        context=context,
        q=q,
    )

    plant.SetVelocities(
        context=context,
        v=qd,
    )

    # Compute the mass matrix:
    M = plant.CalcMassMatrix(
        context=context,
    )

    #  Compute bias terms:
    C = plant.CalcBiasTerm(
        context=context,
    )

    # Compute generalized forces due to gravity:
    tau_g = plant.CalcGravityGeneralizedForces(
        context=context,
    )

    return M, C, tau_g, plant, context
