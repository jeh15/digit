import numpy as np
import numpy.typing as npt

# Types:
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Context
from pydrake.math import RigidTransform
from pydrake.multibody.tree import JacobianWrtVariable, Frame


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
        frame_A=frame_base_body,
        frame_B=frame_body,
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
        frame_A=frame_base_body,
        frame_B=frame_body,
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


# def calculate_kinematic_constraints(
#     plant: MultibodyPlant,
#     context: Context,
#     auxiliary_frames: dict,
#     q: npt.ArrayLike,
#     qd: npt.ArrayLike,
# ) -> [np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute the task space transform, jacobian of the position vector in task space, and bias terms.

#     Args:
#         plant: The MultibodyPlant.
#         context: Context of the plant.
#         q: Generalized positions.
#         qd: Generalized velocities.

#     Returns:
#         constraint_jacobian: The jacobian of the kinematic constraints.
#         con_bias: The bias terms of the kinematic constraints.

#     """
#     # Update plant model to the current state:
#     plant.SetPositions(
#         context=context,
#         q=q,
#     )
#     plant.SetVelocities(
#         context=context,
#         v=qd,
#     )

#     # Zip frames:
#     frames = [
#         # (auxiliary_frames["left_achilles_rod"]["spring_frame"], auxiliary_frames["left_achilles_rod"]["hip_frame"]),
#         # (auxiliary_frames["left_toe_a"]["roll_frame"], auxiliary_frames["left_toe_a"]["motor_frame"]),
#         # (auxiliary_frames["left_toe_b"]["roll_frame"], auxiliary_frames["left_toe_b"]["motor_frame"]),
#         (auxiliary_frames["right_achilles_rod"]["spring_frame"], auxiliary_frames["right_achilles_rod"]["hip_frame"]),
#         (auxiliary_frames["right_toe_a"]["roll_frame"], auxiliary_frames["right_toe_a"]["motor_frame"]),
#         (auxiliary_frames["right_toe_b"]["roll_frame"], auxiliary_frames["right_toe_b"]["motor_frame"]),
#     ]

#     frame_E = plant.world_frame()

#     constraint_jacobian = []
#     constraint_bias = []
#     for frame_A, frame_B in frames:
#         constraint_jacobian.append(
#             plant.CalcJacobianSpatialVelocity(
#                 context=context,
#                 with_respect_to=JacobianWrtVariable.kV,
#                 frame_B=frame_B,
#                 p_BoBp_B=np.zeros((3, 1)),
#                 frame_A=frame_A,
#                 frame_E=frame_E,
#             )
#         )

#         # Calculate the bias terms in task space:
#         constraint_bias.append(
#             plant.CalcBiasSpatialAcceleration(
#                 context=context,
#                 with_respect_to=JacobianWrtVariable.kV,
#                 frame_B=frame_B,
#                 p_BoBp_B=np.zeros((3, 1)),
#                 frame_A=frame_A,
#                 frame_E=frame_E,
#             ).get_coeffs()
#         )

#     constraint_jacobian = np.vstack(constraint_jacobian)
#     constraint_bias = np.concatenate(constraint_bias)

#     return (
#         constraint_jacobian,
#         constraint_bias,
#     )


def calculate_kinematic_constraints(
    plant: MultibodyPlant,
    context: Context,
    constraint_frames: list,
    q: npt.ArrayLike,
    qd: npt.ArrayLike,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the task space transform, jacobian of the position vector in task space, and bias terms.

    Args:
        plant: The MultibodyPlant.
        context: Context of the plant.
        q: Generalized positions.
        qd: Generalized velocities.

    Returns:
        constraint_jacobian: The jacobian of the kinematic constraints.
        con_bias: The bias terms of the kinematic constraints.

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

    constraint_jacobian = []
    constraint_bias = []
    for frame_A, frame_B in constraint_frames:
        # Calculate the kinematic constraint jacobian w.r.t spatial velocity:
        J, position, J_relative = kinematic_constraint_jacobian(
            plant=plant,
            context=context,
            frame_A=frame_A,
            frame_B=frame_B,
        )

        # Calculate the bias terms: dJ * v
        bias = kinematic_constraint_bias(
            plant=plant,
            context=context,
            frame_A=frame_A,
            frame_B=frame_B,
            constraint_jacobian=J,
            relative_position=position,
            relative_jacobian=J_relative,
        )

        constraint_jacobian.append(J)
        constraint_bias.append(bias)

    constraint_jacobian = np.vstack(constraint_jacobian)
    constraint_bias = np.concatenate(constraint_bias)

    return constraint_jacobian, constraint_bias


def kinematic_constraint_jacobian(
    plant: MultibodyPlant,
    context: Context,
    frame_A: Frame,
    frame_B: Frame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the jacobian of the kinematic constraints.

    Args:
        context: Context of the plant.
        frame_A: Frame A.
        frame_B: Frame B.

    Returns:
        constraint_jacobian: The jacobian of the kinematic constraints.

    """
    # Relative to World Frame:
    frame_E = plant.world_frame()

    # Point Position Method:
    point_A = plant.CalcPointsPositions(
            context=context,
            frame_B=frame_A,
            p_BQi=np.zeros((3, 1)),
            frame_A=frame_E,
    )

    point_B = plant.CalcPointsPositions(
            context=context,
            frame_B=frame_B,
            p_BQi=np.zeros((3, 1)),
            frame_A=frame_E,
    )

    relative_position = point_A - point_B

    jacobian_A = plant.CalcJacobianTranslationalVelocity(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_A,
        p_BoBi_B=np.zeros((3, 1)),
        frame_A=frame_E,
        frame_E=frame_E,
    )

    jacobian_B = plant.CalcJacobianTranslationalVelocity(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_B,
        p_BoBi_B=np.zeros((3, 1)),
        frame_A=frame_E,
        frame_E=frame_E,
    )

    relative_jacobian = jacobian_A - jacobian_B
    constraint_jacobian = (
        (relative_position.T @ relative_jacobian) / np.linalg.norm(relative_position)
    )

    return constraint_jacobian, relative_position, relative_jacobian


def kinematic_constraint_bias(
    plant: MultibodyPlant,
    context: Context,
    frame_A: Frame,
    frame_B: Frame,
    constraint_jacobian: np.ndarray,
    relative_position: np.ndarray,
    relative_jacobian: np.ndarray,
) -> np.ndarray:
    """
    Compute the bias terms of the kinematic constraints.

    Args:
        context: Context of the plant.
        frame_A: Frame A.
        frame_B: Frame B.

    Returns:
        constraint_bias: The bias terms of the kinematic constraints.

    """
    # Relative to World Frame:
    frame_E = plant.world_frame()

    # Bias Acceleration Terms:
    bias_A = plant.CalcBiasTranslationalAcceleration(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_A,
        p_BoBi_B=np.zeros((3, 1)),
        frame_A=frame_E,
        frame_E=frame_E,
    )

    bias_B = plant.CalcBiasTranslationalAcceleration(
        context=context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_B,
        p_BoBi_B=np.zeros((3, 1)),
        frame_A=frame_E,
        frame_E=frame_E,
    )

    relative_bias = bias_A - bias_B

    position_norm = np.linalg.norm(relative_position)

    jacobian_dot_velocity = np.dot(
        constraint_jacobian, plant.GetVelocities(context=context),
    )

    relative_jacobian_velocity = (
        relative_jacobian @ plant.GetVelocities(context=context)
    )

    constraint_bias = (
            np.linalg.norm(relative_jacobian_velocity) / position_norm
        ) + (
            np.dot(
                relative_position.T, relative_bias
            ) / position_norm
        ) - (
            jacobian_dot_velocity * np.dot(
                relative_position.T, relative_jacobian_velocity
            ) / (position_norm ** 2)
        )

    return constraint_bias.flatten()
