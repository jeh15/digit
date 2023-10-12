import numpy as np
from pydrake.multibody.plant import MultibodyPlant


def apply_kinematic_constraints(plant: MultibodyPlant):
    # Add SAP Constraints for closed kinematic chains:
    achilles_distance = 0.5
    left_heel_spring_frame = plant.GetFrameByName("left-heel-spring_link")
    left_hip_frame = plant.GetFrameByName("left-hip-pitch_link")
    left_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    left_heel_spring_position = np.array([[0.113789, -0.011056, 0]]).T
    stiffness = 1e6
    damping = 2e3
    # Left Achilles Rod:
    plant.AddDistanceConstraint(
        left_hip_frame.body(),
        left_hip_position,
        left_heel_spring_frame.body(),
        left_heel_spring_position,
        achilles_distance,
        stiffness,
        damping,
    )

    # Right Achilles Rod:
    right_hip_frame = plant.GetFrameByName("right-hip-pitch_link")
    right_heel_spring_frame = plant.GetFrameByName("right-heel-spring_link")
    right_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    right_heel_spring_position = np.array([[0.1, -0.01, 0]]).T
    plant.AddDistanceConstraint(
        right_hip_frame.body(),
        right_hip_position,
        right_heel_spring_frame.body(),
        right_heel_spring_position,
        achilles_distance,
        stiffness,
        damping,
    )

    # Feet Rods:

    # Left Toe A rod:
    rod_a_distance = 0.34
    left_toe_roll_frame = plant.GetFrameByName("left-toe-roll_link")
    left_toe_a_frame = plant.GetFrameByName("left-toe-A_link")
    left_toe_a_roll_position = np.array([[0.0179, -0.009551, -0.054164]]).T
    left_toe_a_position = np.array([[0.057, 0, -0.008]]).T
    plant.AddDistanceConstraint(
        left_toe_a_frame.body(),
        left_toe_a_position,
        left_toe_roll_frame.body(),
        left_toe_a_roll_position,
        rod_a_distance,
        stiffness,
        damping,
    )

    # Left Toe B rod:
    rod_b_distance = 0.288
    left_toe_b_frame = plant.GetFrameByName("left-toe-B_link")
    left_toe_b_roll_position = np.array([[-0.0181, -0.009551, -0.054164]]).T
    left_toe_b_position = np.array([[0.057, 0, -0.008]]).T
    plant.AddDistanceConstraint(
        left_toe_b_frame.body(),
        left_toe_b_position,
        left_toe_roll_frame.body(),
        left_toe_b_roll_position,
        rod_b_distance,
        stiffness,
        damping,
    )

    # Right Toe A rod:
    right_toe_roll_frame = plant.GetFrameByName("right-toe-roll_link")
    right_toe_a_frame = plant.GetFrameByName("right-toe-A_link")
    right_toe_a_roll_position = np.array([[0.0179, 0.009551, -0.054164]]).T
    right_toe_a_position = np.array([[0.057, 0, -0.008]]).T
    plant.AddDistanceConstraint(
        right_toe_a_frame.body(),
        right_toe_a_position,
        right_toe_roll_frame.body(),
        right_toe_a_roll_position,
        rod_a_distance,
        stiffness,
        damping,
    )

    # Right Toe B rod:
    right_toe_b_frame = plant.GetFrameByName("right-toe-B_link")
    right_toe_b_roll_position = np.array([[-0.0181, 0.009551, -0.054164]]).T
    right_toe_b_position = np.array([[0.057, 0, -0.008]]).T
    plant.AddDistanceConstraint(
        right_toe_b_frame.body(),
        right_toe_b_position,
        right_toe_roll_frame.body(),
        right_toe_b_roll_position,
        rod_b_distance,
        stiffness,
        damping,
    )

    return plant
