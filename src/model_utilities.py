import numpy as np
from pydrake.multibody.plant import MultibodyPlant, CoulombFriction
from pydrake.math import RigidTransform
from pydrake.geometry import HalfSpace


def apply_kinematic_constraints(plant: MultibodyPlant) -> MultibodyPlant:
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


def add_terrian(
    plant: MultibodyPlant,
    mu_static: float = 0.5,
    mu_dynamic: float = 0.5,
) -> MultibodyPlant:
    """
    Add a flat ground plane to the plant.

    Utility function adapted from DAIRLab's dairlib.

    Source:
    https://github.com/DAIRLab/dairlib/blob/f2bc1ae1ae7d45f4b8ba731480eac3ce484b2b09/multibody/multibody_utils.cc#L122
    """
    friction = CoulombFriction(
        static_friction=mu_static, dynamic_friction=mu_dynamic,
    )
    halfspace = HalfSpace().MakePose(
        Hz_dir_F=np.array([0, 0, 1]), p_FB=np.array([0, 0, 0]),
    )
    transform = RigidTransform(
        halfspace,
    )
    plant.RegisterCollisionGeometry(
        body=plant.world_body(),
        X_BG=transform,
        shape=HalfSpace(),
        name="ground_collision",
        coulomb_friction=friction,
    )
    plant.RegisterVisualGeometry(
        body=plant.world_body(),
        X_BG=transform,
        shape=HalfSpace(),
        name="ground_visual",
        diffuse_color=[0.5, 0.5, 0.5, 1.0],
    )

    return plant
