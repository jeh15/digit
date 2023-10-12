import numpy as np
import numpy.typing as npt

# Types:
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Context
from pydrake.math import RigidTransform
from pydrake.multibody.tree import JacobianWrtVariable


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


def get_transform(
    plant: MultibodyPlant,
    context: Context,
    body_name: str,
    base_body_name: str,
    q: npt.ArrayLike,
) -> tuple[RigidTransform, MultibodyPlant, Context]:
    """
    Computes the transform of a body (body) relative to
    a base body (base_body).

    Args:
        plant: The MultibodyPlant.
        context: Context of the plant.
        body_name: Name of body.
        base_body_name: Name of the base body to calculate the
                        transform with respects to.
        q: Generalized positions.

    Returns:
        transform_object: The transform object containing the
                          rotation matrix and translation of
                          body relative to base body.
        plant: updated plant
        context: updated context

    """
    # Update plant model to the current state:
    plant.SetPositions(
        context=context,
        q=q,
    )

    # Get frame of body and base body:
    frame_body = plant.GetFrameByName(body_name)
    frame_base_body = plant.GetFrameByName(base_body_name)

    transform_object = plant.CalcRelativeTransform(
        context=context,
        frame_A=frame_body,
        frame_B=frame_base_body,
    )

    return transform_object, plant, context


def calculate_task_space_matricies(
    plant: MultibodyPlant,
    context: Context,
    body_name: str,
    base_body_name: str,
    q: npt.ArrayLike,
    qd: npt.ArrayLike,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the task space transform, jacobian of the position vector in task space, and bias terms.

    Args:
        plant: The MultibodyPlant.
        context: Context of the plant.
        body_name: Name of body.
        base_body_name: Name of the base body to calculate the
                        transform with respects to.
        q: Generalized positions.
        qd: Generalized velocities.

    Returns:
        task_space_transform: The transform of the body relative to the base body.
        task_space_jacobian: The jacobian of the position vector
                             in task space.
        task_space_bias_terms: The bias terms in task space.

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

    # Get frame of body and base body:
    frame_body = plant.GetFrameByName(body_name)
    frame_base_body = plant.GetFrameByName(base_body_name)
    frame_world = plant.world_frame()

    # Calculate the task space transform:
    task_space_transform = plant.CalcRelativeTransform(
        context=context,
        frame_A=frame_body,
        frame_B=frame_base_body,
    )

    # Calculate the jacobian of the position vector in task space:
    spatial_velocity_jacobian = plant.CalcJacobianSpatialVelocity(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_body,
        p_BoBp_B=np.zeros((3, 1)),
        frame_A=frame_base_body,
        frame_E=frame_base_body,
    )

    # Calculate the bias terms in task space:
    bias_spatial_acceleration = plant.CalcBiasSpatialAcceleration(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_body,
        p_BoBp_B=np.zeros((3, 1)),
        frame_A=frame_base_body,
        frame_E=frame_base_body,
    ).get_coeffs()

    return (
        task_space_transform,
        spatial_velocity_jacobian,
        bias_spatial_acceleration,
    )
