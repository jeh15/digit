import os
import subprocess
from absl import app

import casadi

# Typing:
from casadi import MX


def equality_constraints(
    q: MX,
    M: MX,
    C: MX,
    tau_g: MX,
    B: MX,
    H: MX,
    H_bias: MX,
    J: MX,
    J_bias: MX,
) -> MX:
    """Equality constraints for the dynamics of a system.

    Args:
        q: design vector.
        M: The mass matrix.
        C: The Coriolis matrix.
        tau_g: The gravity vector.
        B: The actuation matrix.
        H: The jacobian of the kinematic constraint.
        H_bias: The bias term of the kinematic constraint.
        J: The spatial velocity jacobian.
        J_bias: The bias term of the spatial velocity jacobian.
        split_indx: The index at which the optimization
            variables are split in dv and u.

    Returns:
        The equality constraints.

        Dynamics:
        M @ dv + C @ v - tau_g - B @ u - H.T @ f - J.T @ z = 0

        Holonomic Constraints: (Maybe make this an inequality constraint?)
        H @ dv + H_bias = 0
    """
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, slack_indx = (34, 54, 60, 72, 114)
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    slack = q[z_indx:slack_indx]

    # Extract feet jacobian:
    # [base, left_foot, right_foot, left_hand, right_hand, left_elbow, right_elbow]
    split_J = casadi.vertsplit(J, 6)
    left_foot_J, right_foot_J = split_J[1], split_J[2]
    contact_J = casadi.vertcat(left_foot_J, right_foot_J)

    # Dynamic Constraints:
    dynamics = M @ dv + C - tau_g - B @ u - H.T @ f - contact_J.T @ z

    # Kinematic Constraints:
    kinematic = H @ dv + H_bias

    # Task Objective Slack:
    ddx_task = J @ dv + J_bias
    task_slack = ddx_task - slack

    equality_constraints = casadi.vertcat(dynamics, kinematic, task_slack)

    return equality_constraints


def inequality_constraints(
    q: MX,
    z_previous: MX,
) -> MX:
    """Inequality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.
        z_previous: The previous solution for the f_z ground reaction forces.

    Returns:
        The inequality constraints.

    """
    # Calculate inequality constraints:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, slack_indx = (34, 54, 60, 72, 114)
    z = q[f_indx:z_indx]

    friction = 0.6

    # Constraint: |f_x| + |f_y| <= mu * f_z
    constraint_1 = z[3] + z[4] - friction * z[5]
    constraint_2 = -z[3] + z[4] - friction * z[5]
    constraint_3 = z[3] - z[4] - friction * z[5]
    constraint_4 = -z[3] - z[4] - friction * z[5]

    constraint_5 = z[9] + z[10] - friction * z[11]
    constraint_6 = -z[9] + z[10] - friction * z[11]
    constraint_7 = z[9] - z[10] - friction * z[11]
    constraint_8 = -z[9] - z[10] - friction * z[11]

    # Torsional Friction Constraint:
    # |tau_z| <= r_y x f_x -- r_y = 0.05 -> 0.025
    # |tau_z| <= r_x x f_y -- r_x = 0.1 -> 0.05
    # |tau_z| <= r x (mu * f_z)
    r_y = 0.025 / 2.0
    r_x = 0.05 / 2.0
    constraint_9 = z[2] - r_y * friction * z[5]
    constraint_10 = -z[2] - r_y * friction * z[5]
    # constraint_11 = z[2] - r_x * friction * z[5]
    # constraint_12 = -z[2] - r_x * friction * z[5]

    constraint_13 = z[8] - r_y * friction * z[11]
    constraint_14 = -z[8] - r_y * friction * z[11]
    # constraint_15 = z[8] - r_x * friction * z[11]
    # constraint_16 = -z[8] - r_x * friction * z[11]

    # Zero Moment Constraint:
    # -r_y <= tau_x / f_z <= r_y
    # -r_x <= tau_y / f_z <= r_x
    zero_moment_left_x = z[0] / z_previous[0] - r_y
    zero_moment_left_y = z[1] / z_previous[0] - r_x
    zero_moment_right_x = z[6] / z_previous[1] - r_y
    zero_moment_right_y = z[7] / z_previous[1] - r_x
    zero_moment = casadi.vertcat(
        zero_moment_left_x,
        zero_moment_left_y,
        zero_moment_right_x,
        zero_moment_right_y,
    )

    inequality_constraints = casadi.vertcat(
        constraint_1,
        constraint_2,
        constraint_3,
        constraint_4,
        constraint_5,
        constraint_6,
        constraint_7,
        constraint_8,
        constraint_9,
        constraint_10,
        constraint_13,
        constraint_14,
        zero_moment,
    )

    return inequality_constraints


def objective(
    q: MX,
    desired_task_acceleration: MX,
) -> MX:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, slack_indx = (34, 54, 60, 72, 114)
    dv = q[:dv_indx]
    z = q[f_indx:z_indx]
    slack = q[z_indx:slack_indx]

    # Reshape for easier indexing:
    z = casadi.reshape(z, 2, 6)

    # Split the Task Space Jacobians:
    split_slack = casadi.vertsplit(slack, 6)
    ddx_base = split_slack[0]
    ddx_left_foot, ddx_right_foot = split_slack[1], split_slack[2]
    ddx_left_hand, ddx_right_hand = split_slack[3], split_slack[4]
    ddx_left_elbow, ddx_right_elbow = split_slack[5], split_slack[6]

    split_desired = casadi.vertsplit(desired_task_acceleration, 6)
    desired_base = split_desired[0]
    desired_left_foot, desired_right_foot = split_desired[1], split_desired[2]
    desired_left_hand, desired_right_hand = split_desired[3], split_desired[4]
    desired_left_elbow, desired_right_elbow = split_desired[5], split_desired[6]

    base_tracking_w_weight = 10.0
    base_tracking_x_weight = 10.0
    foot_tracking_w_weight = 100.0
    foot_tracking_x_weight = 100.0
    hand_tracking_w_weight = 10.0
    hand_tracking_x_weight = 10.0
    elbow_tracking_w_weight = 10.0
    elbow_tracking_x_weight = 10.0
    base_error_w = base_tracking_w_weight * (ddx_base[:3] - desired_base[:3]) ** 2
    base_error_x = base_tracking_x_weight * (ddx_base[3:] - desired_base[3:]) ** 2
    left_foot_error_w = foot_tracking_w_weight * (ddx_left_foot[:3] - desired_left_foot[:3]) ** 2
    left_foot_error_x = foot_tracking_x_weight * (ddx_left_foot[3:] - desired_left_foot[3:]) ** 2
    right_foot_error_w = foot_tracking_w_weight * (ddx_right_foot[:3] - desired_right_foot[:3]) ** 2
    right_foot_error_x = foot_tracking_x_weight * (ddx_right_foot[3:] - desired_right_foot[3:]) ** 2
    left_hand_error_w = hand_tracking_w_weight * (ddx_left_hand[:3] - desired_left_hand[:3]) ** 2
    left_hand_error_x = hand_tracking_x_weight * (ddx_left_hand[3:] - desired_left_hand[3:]) ** 2
    right_hand_error_w = hand_tracking_w_weight * (ddx_right_hand[:3] - desired_right_hand[:3]) ** 2
    right_hand_error_x = hand_tracking_x_weight * (ddx_right_hand[3:] - desired_right_hand[3:]) ** 2
    left_elbow_error_w = elbow_tracking_w_weight * (ddx_left_elbow[:3] - desired_left_elbow[:3]) ** 2
    left_elbow_error_x = elbow_tracking_x_weight * (ddx_left_elbow[3:] - desired_left_elbow[3:]) ** 2
    right_elbow_error_w = elbow_tracking_w_weight * (ddx_right_elbow[:3] - desired_right_elbow[:3]) ** 2
    right_elbow_error_x = elbow_tracking_x_weight * (ddx_right_elbow[3:] - desired_right_elbow[3:]) ** 2

    task_objective = casadi.sum1(
        (
            base_error_w
            + base_error_x
            + left_foot_error_w
            + left_foot_error_x
            + right_foot_error_w
            + right_foot_error_x
            + left_hand_error_w
            + left_hand_error_x
            + right_hand_error_w
            + right_hand_error_x
            + left_elbow_error_w
            + left_elbow_error_x
            + right_elbow_error_w
            + right_elbow_error_x
        ),
    )

    # Minimize Arm Movement:
    # Left Arm: 16, 17, 18, 19
    # Right Arm: 30, 31, 32, 33
    left_arm_dv = dv[16:20]
    right_arm_dv = dv[30:34]
    arm_movement = (
        casadi.sum1(left_arm_dv ** 2)
        + casadi.sum1(right_arm_dv ** 2)
    )

    regularization_objective = casadi.sum1(q ** 2)

    task_weight = 1.0
    arm_movement_weight = 1.0
    regularization_weight = 1e-4
    objective_value = (
        task_weight * task_objective
        + arm_movement_weight * arm_movement
        + regularization_weight * regularization_objective
    )

    return objective_value


def main(argv=None):
    dv_size = 34
    u_size = 20
    f_size = 6
    z_size = 12
    slack_size = 42
    spatial_vector_size = 6
    num_task_frames = 7

    dv_indx = dv_size
    u_indx = dv_indx + u_size
    f_indx = u_indx + f_size
    z_indx = f_indx + z_size
    slack_indx = z_indx + slack_size
    split_indx = (dv_indx, u_indx, f_indx, z_indx, slack_indx)

    # Define symbolic variables:
    dv = casadi.MX.sym("dv", dv_size)
    u = casadi.MX.sym("u", u_size)
    f = casadi.MX.sym("f", f_size)
    z = casadi.MX.sym("z", z_size)
    slack = casadi.MX.sym("slack", slack_size)

    design_vector = casadi.vertcat(dv, u, f, z, slack)

    M_matrix = casadi.MX.sym("M", dv_size, dv_size)
    C_matrix = casadi.MX.sym("C", dv_size)
    tau_g_vector = casadi.MX.sym("tau_g", dv_size)
    B_matrix = casadi.MX.sym("B", dv_size, u_size)
    H_matrix = casadi.MX.sym("H", f_size, dv_size)
    H_bias_vector = casadi.MX.sym("H_bias", f_size)
    J_matrix = casadi.MX.sym("J", spatial_vector_size * num_task_frames, dv_size)
    J_bias_matrix = casadi.MX.sym("task_bias", spatial_vector_size * num_task_frames)
    ground_reaction_matrix = casadi.MX.sym("z_previous", 2)
    desired_task_acceleration_matrix = casadi.MX.sym("desired_ddx", spatial_vector_size * num_task_frames)

    equality_constraint_input = [
        design_vector,
        M_matrix,
        C_matrix,
        tau_g_vector,
        B_matrix,
        H_matrix,
        H_bias_vector,
        J_matrix,
        J_bias_matrix,
    ]

    inequality_constraint_input = [
        design_vector,
        ground_reaction_matrix,
    ]

    objective_input = [
        design_vector,
        desired_task_acceleration_matrix,
    ]

    # Convert to CasADi Function:
    b_eq_function = casadi.Function(
        "b_eq_function",
        equality_constraint_input,
        [-equality_constraints(*equality_constraint_input)],
    )

    A_eq_function = casadi.Function(
        "A_eq_function",
        equality_constraint_input,
        [casadi.jacobian(
            equality_constraints(*equality_constraint_input),
            design_vector,
        )],
    )

    b_ineq_function = casadi.Function(
        "b_ineq_function",
        inequality_constraint_input,
        [-inequality_constraints(*inequality_constraint_input)],
    )

    A_ineq_function = casadi.Function(
        "A_ineq_function",
        inequality_constraint_input,
        [casadi.jacobian(
            inequality_constraints(*inequality_constraint_input),
            design_vector,
        )],
    )

    hessian, gradient = casadi.hessian(
        objective(*objective_input),
        design_vector,
    )

    f_function = casadi.Function(
        "f_function",
        objective_input,
        [gradient],
    )

    H_function = casadi.Function(
        "H_function",
        objective_input,
        [hessian],
    )

    # Generate C++ Code:
    opts = {
        "cpp": True,
    }
    filenames = [
        "equality_constraint_function.cpp",
        "inequality_constraint_function.cpp",
        "objective_function.cpp",
    ]
    casadi_functions = [
        [b_eq_function, A_eq_function],
        [b_ineq_function, A_ineq_function],
        [f_function, H_function],
    ]
    loop_iterables = zip(
        filenames,
        casadi_functions,
    )

    for filename, casadi_function in loop_iterables:
        generator = casadi.CodeGenerator(filename, opts)
        for function in casadi_function:
            generator.add(function)
        generator.generate()

    # Compile C++ Code:
    directory_path = os.path.dirname(__file__)
    for filename in filenames:
        compile_cmd = [
            "g++",
            "-fPIC",
            "-shared",
            "-O3",
            "-o",
        ]
        file_path = os.path.join(
            directory_path, filename.replace(".cpp", ".so"),
        )
        compile_cmd.append(file_path)
        compile_cmd.append(filename)
        subprocess.call(compile_cmd)


if __name__ == "__main__":
    app.run(main)
