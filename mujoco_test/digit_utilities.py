import numpy as np


class DigitUtilities():
    """
    Convenience class for Digit index mappings.
    """
    def __init__(self):
        self.actuation_idx = {
            "left_leg": [0, 1, 2, 3, 4, 5],
            "left_arm": [6, 7, 8, 9],
            "right_leg": [10, 11, 12, 13, 14, 15],
            "right_arm": [16, 17, 18, 19],
        }
        self.actuated_joints_idx = {
            "left_leg": [0, 1, 2, 3, 6, 7],
            "left_arm": [10, 11, 12, 13],
            "right_leg": [14, 15, 16, 17, 20, 21],
            "right_arm": [24, 25, 26, 27],
        }

        # Left Mappings:
        self.left_hip_roll = {
            "actuation_idx": 0,
            "joint_idx": 0,
        }
        self.left_hip_yaw = {
            "actuation_idx": 1,
            "joint_idx": 1,
        }
        self.left_hip_pitch = {
            "actuation_idx": 2,
            "joint_idx": 2,
        }
        self.left_knee = {
            "actuation_idx": 3,
            "joint_idx": 3,
        }
        self.left_toe_a = {
            "actuation_idx": 4,
            "joint_idx": 6,
        }
        self.left_toe_b = {
            "actuation_idx": 5,
            "joint_idx": 7,
        }
        self.left_shoulder_roll = {
            "actuation_idx": 6,
            "joint_idx": 10,
        }
        self.left_shoulder_pitch = {
            "actuation_idx": 7,
            "joint_idx": 11,
        }
        self.left_shoulder_yaw = {
            "actuation_idx": 8,
            "joint_idx": 12,
        }
        self.left_elbow = {
            "actuation_idx": 9,
            "joint_idx": 13,
        }

        # Right Mappings:
        self.right_hip_roll = {
            "actuation_idx": 10,
            "joint_idx": 14,
        }
        self.right_hip_yaw = {
            "actuation_idx": 11,
            "joint_idx": 15,
        }
        self.right_hip_pitch = {
            "actuation_idx": 12,
            "joint_idx": 16,
        }
        self.right_knee = {
            "actuation_idx": 13,
            "joint_idx": 17,
        }
        self.right_toe_a = {
            "actuation_idx": 14,
            "joint_idx": 20,
        }
        self.right_toe_b = {
            "actuation_idx": 15,
            "joint_idx": 21,
        }
        self.right_shoulder_roll = {
            "actuation_idx": 16,
            "joint_idx": 24,
        }
        self.right_shoulder_pitch = {
            "actuation_idx": 17,
            "joint_idx": 25,
        }
        self.right_shoulder_yaw = {
            "actuation_idx": 18,
            "joint_idx": 26,
        }
        self.right_elbow = {
            "actuation_idx": 19,
            "joint_idx": 27,
        }

    # Get names function to pring out joint names:
    # def get_names()

    # Mujoco to Drake Joint Position and Velocity Mappings:
    def joint_map(self, motor_position, joint_position):
        # Extract Agility Digit Mappings:
        motor_left_leg = motor_position[:6]
        left_tarsus_heel = np.asarray([joint_position[1], joint_position[4]])
        left_toe_pitch_roll = joint_position[2:4]
        motor_right_leg = motor_position[6:12]
        right_tarsus_heel = np.asarray([joint_position[6], joint_position[9]])
        right_toe_pitch_roll = joint_position[7:9]

        q_left_leg = np.concatenate(
            [motor_left_leg[:4], left_tarsus_heel, motor_left_leg[4:], left_toe_pitch_roll],
        )
        q_right_leg = np.concatenate(
            [motor_right_leg[:4], right_tarsus_heel, motor_right_leg[4:], right_toe_pitch_roll],
        )
        q_left_arm = motor_position[12:16]
        q_right_arm = motor_position[16:]

        q = np.concatenate([q_left_leg, q_left_arm, q_right_leg, q_right_arm])

        return q

    # Drake to Mujoco Torque Mappings:
    def actuation_map(self, torque):
        tau = np.concatenate(
            [
                torque[self.actuation_idx["left_leg"]],
                torque[self.actuation_idx["right_leg"]],
                torque[self.actuation_idx["left_arm"]], 
                torque[self.actuation_idx["right_arm"]],
            ]
        )
        return tau