import numpy as np


class DigitUtilities():
    """
    Convenience class for Digit index mappings.
    """
    def __init__(
        self,
        floating_base: bool = False,
    ):
        self.num_motors = 20

        self.actuation_idx = {
            "left_leg": [0, 1, 2, 3, 4, 5],
            "left_arm": [6, 7, 8, 9],
            "right_leg": [10, 11, 12, 13, 14, 15],
            "right_arm": [16, 17, 18, 19],
        }

        actuation_idx = np.concatenate(
            list(
                self.actuation_idx.values()
            ),
            axis=0,
        )

        if floating_base:
            self.num_joints = 34

            # Relative to Velocity Generalized Coordinates:
            base_offset = 6

            # Relative to Position Generalized Coordinates:
            self.base_indx = {
                "rotation": [0, 1, 2, 3],
                "position": [4, 5, 6],
            }

            # Relative to Velocity Generalized Coordinates:
            self.actuated_joints_idx = {
                "left_leg": np.array([0, 1, 2, 3, 6, 7]) + base_offset,
                "left_arm": np.array([10, 11, 12, 13]) + base_offset,
                "right_leg": np.array([14, 15, 16, 17, 20, 21]) + base_offset,
                "right_arm": np.array([24, 25, 26, 27]) + base_offset,
            }

            actuated_joints_idx = np.concatenate(
                list(
                    self.actuated_joints_idx.values()
                ),
                axis=0,
            )

            # Left Mappings:
            self.left_hip_roll = {
                "actuation_idx": 0,
                "joint_idx": 0 + base_offset,
            }
            self.left_hip_yaw = {
                "actuation_idx": 1,
                "joint_idx": 1 + base_offset,
            }
            self.left_hip_pitch = {
                "actuation_idx": 2,
                "joint_idx": 2 + base_offset,
            }
            self.left_knee = {
                "actuation_idx": 3,
                "joint_idx": 3 + base_offset,
            }
            self.left_toe_a = {
                "actuation_idx": 4,
                "joint_idx": 6 + base_offset,
            }
            self.left_toe_b = {
                "actuation_idx": 5,
                "joint_idx": 7 + base_offset,
            }
            self.left_toe_pitch = {
                "actuation_idx": None,
                "joint_idx": 8 + base_offset,
            }
            self.left_toe_roll = {
                "actuation_idx": None,
                "joint_idx": 9 + base_offset,
            }
            self.left_shoulder_roll = {
                "actuation_idx": 6,
                "joint_idx": 10 + base_offset,
            }
            self.left_shoulder_pitch = {
                "actuation_idx": 7,
                "joint_idx": 11 + base_offset,
            }
            self.left_shoulder_yaw = {
                "actuation_idx": 8,
                "joint_idx": 12 + base_offset,
            }
            self.left_elbow = {
                "actuation_idx": 9,
                "joint_idx": 13 + base_offset,
            }

            # Right Mappings:
            self.right_hip_roll = {
                "actuation_idx": 10,
                "joint_idx": 14 + base_offset,
            }
            self.right_hip_yaw = {
                "actuation_idx": 11,
                "joint_idx": 15 + base_offset,
            }
            self.right_hip_pitch = {
                "actuation_idx": 12,
                "joint_idx": 16 + base_offset,
            }
            self.right_knee = {
                "actuation_idx": 13,
                "joint_idx": 17 + base_offset,
            }
            self.right_toe_a = {
                "actuation_idx": 14,
                "joint_idx": 20 + base_offset,
            }
            self.right_toe_b = {
                "actuation_idx": 15,
                "joint_idx": 21 + base_offset,
            }
            self.right_toe_pitch = {
                "actuation_idx": None,
                "joint_idx": 22 + base_offset,
            }
            self.right_toe_roll = {
                "actuation_idx": None,
                "joint_idx": 23 + base_offset,
            }
            self.right_shoulder_roll = {
                "actuation_idx": 16,
                "joint_idx": 24 + base_offset,
            }
            self.right_shoulder_pitch = {
                "actuation_idx": 17,
                "joint_idx": 25 + base_offset,
            }
            self.right_shoulder_yaw = {
                "actuation_idx": 18,
                "joint_idx": 26 + base_offset,
            }
            self.right_elbow = {
                "actuation_idx": 19,
                "joint_idx": 27 + base_offset,
            }

            # Control Input Mapping:
            # gain_1 = 1.0 / 2.0
            # gain_2 = 1.0 / (1.0 * 50.0)
            # gain_3 = 1.0 / (3.0 * 50.0)
            gain_1 = 1.0 / 2.0
            gain_2 = 1.0 / (1.0 * 50.0)
            gain_3 = 1.0 / (3.0 * 50.0)
            self.control_matrix = np.zeros((self.num_joints, self.num_motors))
            self.control_matrix[actuated_joints_idx, actuation_idx] = 1.0

            # Manual Gear Ratio:
            # self.control_matrix[
            #     self.left_hip_roll["joint_idx"], self.left_hip_roll["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.left_hip_yaw["joint_idx"], self.left_hip_yaw["actuation_idx"]
            # ] = 1.0 / 50.0
            # self.control_matrix[
            #     self.left_hip_pitch["joint_idx"], self.left_hip_pitch["actuation_idx"]
            # ] = 1.0 / 16.0
            # self.control_matrix[
            #     self.left_knee["joint_idx"], self.left_knee["actuation_idx"]
            # ] = 1.0 / 16.0
            # self.control_matrix[
            #     self.left_shoulder_roll["joint_idx"], self.left_shoulder_roll["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.left_shoulder_pitch["joint_idx"], self.left_shoulder_pitch["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.left_shoulder_yaw["joint_idx"], self.left_shoulder_yaw["actuation_idx"]
            # ] = 1.0 / 50.0
            # self.control_matrix[
            #     self.left_elbow["joint_idx"], self.left_elbow["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.right_hip_roll["joint_idx"], self.right_hip_roll["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.right_hip_yaw["joint_idx"], self.right_hip_yaw["actuation_idx"]
            # ] = 1.0 / 50.0
            # self.control_matrix[
            #     self.right_hip_pitch["joint_idx"], self.right_hip_pitch["actuation_idx"]
            # ] = 1.0 / 16.0
            # self.control_matrix[
            #     self.right_knee["joint_idx"], self.right_knee["actuation_idx"]
            # ] = 1.0 / 16.0
            # self.control_matrix[
            #     self.right_shoulder_roll["joint_idx"], self.right_shoulder_roll["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.right_shoulder_pitch["joint_idx"], self.right_shoulder_pitch["actuation_idx"]
            # ] = 1.0 / 80.0
            # self.control_matrix[
            #     self.right_shoulder_yaw["joint_idx"], self.right_shoulder_yaw["actuation_idx"]
            # ] = 1.0 / 50.0
            # self.control_matrix[
            #     self.right_elbow["joint_idx"], self.right_elbow["actuation_idx"]
            # ] = 1.0 / 80.0

            # Toe Motor Mappings:
            self.control_matrix[
                self.left_toe_a["joint_idx"], self.left_toe_a["actuation_idx"]
            ] = 1.0 * gain_1
            self.control_matrix[
                self.left_toe_b["joint_idx"], self.left_toe_b["actuation_idx"]
            ] = 1.0 * gain_1
            self.control_matrix[
                self.left_toe_pitch["joint_idx"], self.left_toe_a["actuation_idx"]
            ] = -1.0 * gain_2
            self.control_matrix[
                self.left_toe_pitch["joint_idx"], self.left_toe_b["actuation_idx"]
            ] = 1.0 * gain_2
            self.control_matrix[
                self.left_toe_roll["joint_idx"], self.left_toe_a["actuation_idx"]
            ] = 1.0 * gain_3
            self.control_matrix[
                self.left_toe_roll["joint_idx"], self.left_toe_b["actuation_idx"]
            ] = 1.0 * gain_3
            self.control_matrix[
                self.right_toe_a["joint_idx"], self.right_toe_a["actuation_idx"]
            ] = 1.0 * gain_1
            self.control_matrix[
                self.right_toe_b["joint_idx"], self.right_toe_b["actuation_idx"]
            ] = 1.0 * gain_1
            self.control_matrix[
                self.right_toe_pitch["joint_idx"], self.right_toe_a["actuation_idx"]
            ] = -1.0 * gain_2
            self.control_matrix[
                self.right_toe_pitch["joint_idx"], self.right_toe_b["actuation_idx"]
            ] = 1.0 * gain_2
            self.control_matrix[
                self.right_toe_roll["joint_idx"], self.right_toe_a["actuation_idx"]
            ] = 1.0 * gain_3
            self.control_matrix[
                self.right_toe_roll["joint_idx"], self.right_toe_b["actuation_idx"]
            ] = 1.0 * gain_3

        else:
            self.num_joints = 28

            self.actuated_joints_idx = {
                "left_leg": [0, 1, 2, 3, 6, 7],
                "left_arm": [10, 11, 12, 13],
                "right_leg": [14, 15, 16, 17, 20, 21],
                "right_arm": [24, 25, 26, 27],
            }

            actuated_joints_idx = np.concatenate(
                list(
                    self.actuated_joints_idx.values()
                ),
                axis=0,
            )
            self.control_matrix = np.zeros((self.num_joints, self.num_motors))
            self.control_matrix[actuated_joints_idx, actuation_idx] = 1.0

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

    # Mujoco to Drake Joint Position and Velocity Mappings:
    def joint_map(
        self,
        motor_position,
        joint_position,
        base_position = None,
    ) -> np.ndarray:
        """
            base_position assumes the form of [base_rotation_elements, base_translation_elements]
        """
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

        if base_position is None:
            q = np.concatenate([q_left_leg, q_left_arm, q_right_leg, q_right_arm])
        else:
            q = np.concatenate([base_position, q_left_leg, q_left_arm, q_right_leg, q_right_arm])

        return q

    # Drake to Mujoco Torque Mappings:
    def actuation_map(self, torque) -> np.ndarray:
        tau = np.concatenate(
            [
                torque[self.actuation_idx["left_leg"]],
                torque[self.actuation_idx["right_leg"]],
                torque[self.actuation_idx["left_arm"]],
                torque[self.actuation_idx["right_arm"]],
            ]
        )
        return tau
