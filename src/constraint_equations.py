import numpy as np

def main(argv=None):
    # Calculate the kinematic constraint:

    # For the right achilles:
    right_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    right_heel_spring_position = np.array([[0.1, -0.01, 0]]).T

    hip_transform = plant.CalcPointsPositions(
        context=plant_context,
        frame_B=plant.GetFrameByName("right-hip-pitch_link"),
        p_BQi=right_hip_position,
        frame_A=plant.world_frame(),
    )

    heel_transform = plant.CalcPointsPositions(
        context=plant_context,
        frame_B=plant.GetFrameByName("right-heel-spring_link"),
        p_BQi=right_heel_spring_position,
        frame_A=plant.world_frame(),
    )

    # ddh = H @ dv + dH @ v
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