import numpy as np
from pydrake.multibody.plant import MultibodyPlant, CoulombFriction
from pydrake.multibody.tree import FixedOffsetFrame, WeldJoint
from pydrake.math import RigidTransform
from pydrake.geometry import HalfSpace


def apply_kinematic_constraints(plant: MultibodyPlant):
    # Add SAP Constraints for closed kinematic chains:
    achilles_distance = 0.5
    left_heel_spring_frame = plant.GetFrameByName("left-heel-spring_link")
    left_hip_frame = plant.GetFrameByName("left-hip-pitch_link")
    left_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    left_heel_spring_position = np.array([[0.113789, -0.011056, 0]]).T
    stiffness = 1e6
    damping = 2e3
    # stiffness = 1e8
    # damping = 2e3
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


def add_terrian(
    plant: MultibodyPlant,
    mu_static: float = 0.5,
    mu_dynamic: float = 0.5,
):
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


def add_auxiliary_frames(
    plant: MultibodyPlant,
) -> dict:
    # Left Achilles Rod:
    left_heel_spring_frame = plant.GetFrameByName("left-heel-spring_link")
    left_hip_frame = plant.GetFrameByName("left-hip-pitch_link")
    left_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    left_heel_spring_position = np.array([[0.113789, -0.011056, 0]]).T
    left_achilles_rod_spring_frame = FixedOffsetFrame(
        name="left_achilles_rod_spring_frame",
        P=left_heel_spring_frame,
        X_PF=RigidTransform(
            p=left_heel_spring_position,
        ),
    )
    left_achilles_rod_hip_frame = FixedOffsetFrame(
        name="left_achilles_rod_hip_frame",
        P=left_hip_frame,
        X_PF=RigidTransform(
            p=left_hip_position,
        ),
    )
    plant.AddFrame(left_achilles_rod_spring_frame)
    plant.AddFrame(left_achilles_rod_hip_frame)

    # Right Achilles Rod:
    right_hip_frame = plant.GetFrameByName("right-hip-pitch_link")
    right_heel_spring_frame = plant.GetFrameByName("right-heel-spring_link")
    right_hip_position = np.array([[0.0, 0.0, 0.046]]).T
    right_heel_spring_position = np.array([[0.1, -0.01, 0]]).T
    right_achilles_rod_spring_frame = FixedOffsetFrame(
        name="right_achilles_rod_spring_frame",
        P=right_heel_spring_frame,
        X_PF=RigidTransform(
            p=right_heel_spring_position,
        ),
    )
    right_achilles_rod_hip_frame = FixedOffsetFrame(
        name="right_achilles_rod_hip_frame",
        P=right_hip_frame,
        X_PF=RigidTransform(
            p=right_hip_position,
        ),
    )
    plant.AddFrame(right_achilles_rod_spring_frame)
    plant.AddFrame(right_achilles_rod_hip_frame)

    # Left Toe A rod:
    left_toe_roll_frame = plant.GetFrameByName("left-toe-roll_link")
    left_toe_a_frame = plant.GetFrameByName("left-toe-A_link")
    left_toe_a_roll_position = np.array([[0.0179, -0.009551, -0.054164]]).T
    left_toe_a_position = np.array([[0.057, 0, -0.008]]).T
    left_toe_a_motor_frame = FixedOffsetFrame(
        name="left_toe_a_motor_frame",
        P=left_toe_a_frame,
        X_PF=RigidTransform(
            p=left_toe_a_position,
        ),
    )
    left_toe_a_roll_frame = FixedOffsetFrame(
        name="left_toe_a_roll_frame",
        P=left_toe_roll_frame,
        X_PF=RigidTransform(
            p=left_toe_a_roll_position,
        ),
    )
    plant.AddFrame(left_toe_a_motor_frame)
    plant.AddFrame(left_toe_a_roll_frame)

    # Left Toe B rod:
    left_toe_b_frame = plant.GetFrameByName("left-toe-B_link")
    left_toe_b_roll_position = np.array([[-0.0181, -0.009551, -0.054164]]).T
    left_toe_b_position = np.array([[0.057, 0, -0.008]]).T
    left_toe_b_motor_frame = FixedOffsetFrame(
        name="left_toe_b_motor_frame",
        P=left_toe_b_frame,
        X_PF=RigidTransform(
            p=left_toe_b_position,
        ),
    )
    left_toe_b_roll_frame = FixedOffsetFrame(
        name="left_toe_b_roll_frame",
        P=left_toe_roll_frame,
        X_PF=RigidTransform(
            p=left_toe_b_roll_position,
        ),
    )
    plant.AddFrame(left_toe_b_motor_frame)
    plant.AddFrame(left_toe_b_roll_frame)

    # Right Toe A rod:
    right_toe_roll_frame = plant.GetFrameByName("right-toe-roll_link")
    right_toe_a_frame = plant.GetFrameByName("right-toe-A_link")
    right_toe_a_roll_position = np.array([[0.0179, 0.009551, -0.054164]]).T
    right_toe_a_position = np.array([[0.057, 0, -0.008]]).T
    right_toe_a_motor_frame = FixedOffsetFrame(
        name="right_toe_a_motor_frame",
        P=right_toe_a_frame,
        X_PF=RigidTransform(
            p=right_toe_a_position,
        ),
    )
    right_toe_a_roll_frame = FixedOffsetFrame(
        name="right_toe_a_roll_frame",
        P=right_toe_roll_frame,
        X_PF=RigidTransform(
            p=right_toe_a_roll_position,
        ),
    )
    plant.AddFrame(right_toe_a_motor_frame)
    plant.AddFrame(right_toe_a_roll_frame)

    # Right Toe B rod:
    right_toe_b_frame = plant.GetFrameByName("right-toe-B_link")
    right_toe_b_roll_position = np.array([[-0.0181, 0.009551, -0.054164]]).T
    right_toe_b_position = np.array([[0.057, 0, -0.008]]).T
    right_toe_b_motor_frame = FixedOffsetFrame(
        name="right_toe_b_motor_frame",
        P=right_toe_b_frame,
        X_PF=RigidTransform(
            p=right_toe_b_position,
        ),
    )
    right_toe_b_roll_frame = FixedOffsetFrame(
        name="right_toe_b_roll_frame",
        P=right_toe_roll_frame,
        X_PF=RigidTransform(
            p=right_toe_b_roll_position,
        ),
    )
    plant.AddFrame(right_toe_b_motor_frame)
    plant.AddFrame(right_toe_b_roll_frame)

    auxiliary_frames = {
        "left_achilles_rod": {
            "spring_frame": left_achilles_rod_spring_frame,
            "hip_frame": left_achilles_rod_hip_frame,
        },
        "right_achilles_rod": {
            "spring_frame": right_achilles_rod_spring_frame,
            "hip_frame": right_achilles_rod_hip_frame,
        },
        "left_toe_a": {
            "motor_frame": left_toe_a_motor_frame,
            "roll_frame": left_toe_a_roll_frame,
        },
        "left_toe_b": {
            "motor_frame": left_toe_b_motor_frame,
            "roll_frame": left_toe_b_roll_frame,
        },
        "right_toe_a": {
            "motor_frame": right_toe_a_motor_frame,
            "roll_frame": right_toe_a_roll_frame,
        },
        "right_toe_b": {
            "motor_frame": right_toe_b_motor_frame,
            "roll_frame": right_toe_b_roll_frame,
        },
    }

    return auxiliary_frames


def teststand_configuration(
    plant: MultibodyPlant,
):  
    # CANT USE THIS METHOD... ADDS NEW JOINTS
    base_frame = plant.GetFrameByName("base_link")
    WeldJoint(
        "base_to_world",
        plant.world_frame(),
        base_frame,
        X_FM=RigidTransform(
            p=np.array([0.0, 0.0, 1.2]),
        ),
    )