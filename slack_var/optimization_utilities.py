from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from pydrake.solvers import (
    MathematicalProgram, 
    Solve,
    QuadraticCost,
    LinearConstraint,
)


# Type Annotations:
from osqp import OSQP
Solution = Any

jax.config.update("jax_enable_x64", True)

# Surpress warnings:
# ruff: noqa: E731
# flake8: noqa: E731

significant_bits = 2
eps = np.finfo(np.float64).eps
# eps = 1e-4


def condition_matrix(matrices):
    condition_matrix = []
    for matrix in matrices:
        matrix = np.asarray(matrix)
        matrix = np.where(np.abs(matrix) < significant_bits * eps, 0.0, matrix)
        condition_matrix.append(matrix)

    return tuple(condition_matrix)


@partial(jax.jit, static_argnames=['split_indx'])
def equality_constraints(
    q: jax.typing.ArrayLike,
    M: jax.typing.ArrayLike,
    C: jax.typing.ArrayLike,
    tau_g: jax.typing.ArrayLike,
    B: jax.typing.ArrayLike,
    H: jax.typing.ArrayLike,
    H_bias: jax.typing.ArrayLike,
    J: jax.typing.ArrayLike,
    J_bias: jax.typing.ArrayLike,
    split_indx: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    """Equality constraints for the dynamics of a system.

    Args:
        q: Spatial velocities and constraint forces.
        M: The mass matrix.
        C: The Coriolis matrix.
        tau_g: The gravity vector.
        B: The actuation matrix.
        H: The jacobian of the kinematic constraint.
        H_bias: The bias term of the kinematic constraint.
        J: The spatial velocity jacobian.
        z_previous: The previous solution for the f_z ground reaction forces.
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
    dv_indx, u_indx, f_indx, z_indx, slack_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    slack = q[z_indx:slack_indx]

    # Extract feet jacobian:
    # [base, left_foot, right_foot, left_hand, right_hand, left_elbow, right_elbow]
    split_J = jnp.split(J, 7)
    left_foot_J, right_foot_J = split_J[1], split_J[2]
    contact_J = jnp.concatenate(
        [left_foot_J, right_foot_J],
    )

    # Dynamic Constraints:
    dynamics = M @ dv + C - tau_g - B @ u - H.T @ f - contact_J.T @ z

    # Kinematic Constraints:
    kinematic = H @ dv + H_bias

    # Task Objective Slack:
    ddx_task = J @ dv + J_bias
    task_slack = ddx_task - slack

    equality_constraints = jnp.concatenate(
        [
            dynamics,
            kinematic,
            task_slack,
        ],
    )

    return equality_constraints


@partial(jax.jit, static_argnames=['friction', 'split_indx'])
def inequality_constraints(
    q: jax.typing.ArrayLike,
    z_previous: jax.typing.ArrayLike,
    friction: float,
    split_indx: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    """Inequality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.
        z_previous: The previous solution for the f_z ground reaction forces.

    Returns:
        The inequality constraints.

    """
    # Calculate inequality constraints:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, slack_indx = split_indx
    z = q[f_indx:z_indx]

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))

    # Constraint: |f_x| + |f_y| <= mu * f_z
    constraint_1 = z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_2 = -z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_3 = z[:, -3] - z[:, -2] - friction * z[:, -1]
    constraint_4 = -z[:, -3] - z[:, -2] - friction * z[:, -1]

    # Torsional Friction Constraint:
    # |tau_z| <= r_y x f_x -- r_y = 0.05 -> 0.025
    # |tau_z| <= r_x x f_y -- r_x = 0.1 -> 0.05
    # |tau_z| <= r x (mu * f_z)
    r_y = 0.025 / 2.0
    r_x = 0.05 / 2.0
    constraint_5 = z[:, 2] - r_y * friction * z[:, -1]
    constraint_6 = -z[:, 2] - r_y * friction * z[:, -1]
    constraint_7 = z[:, 2] - r_x * friction * z[:, -1]
    constraint_8 = -z[:, 2] - r_x * friction * z[:, -1]

    # Zero Moment Constraint:
    # -r_y <= tau_x / f_z <= r_y
    # -r_x <= tau_y / f_z <= r_x
    zero_moment_left_x = z[0, 0] / z_previous[0] - r_y
    zero_moment_left_y = z[0, 1] / z_previous[0] - r_x
    zero_moment_right_x = z[1, 0] / z_previous[1] - r_y
    zero_moment_right_y = z[1, 1] / z_previous[1] - r_x
    zero_moment = jnp.array([zero_moment_left_x, zero_moment_left_y, zero_moment_right_x, zero_moment_right_y])
    
    inequality_constraints = jnp.concatenate(
        [
            constraint_1,
            constraint_2,
            constraint_3,
            constraint_4,
            constraint_5,
            constraint_6,
            constraint_7,
            constraint_8,
            zero_moment,
        ],
    )

    return inequality_constraints


@partial(jax.jit, static_argnames=['split_indx'])
def objective(
    q: jax.typing.ArrayLike,
    desired_task_acceleration: jax.typing.ArrayLike,
    split_indx: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, slack_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    slack = q[z_indx:slack_indx]

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))

    # Split the Task Space Jacobians:
    split_slack = jnp.split(slack, 7)
    ddx_base = split_slack[0]
    ddx_left_foot, ddx_right_foot = split_slack[1], split_slack[2]
    ddx_left_hand, ddx_right_hand = split_slack[3], split_slack[4]
    ddx_left_elbow, ddx_right_elbow = split_slack[5], split_slack[6]

    split_desired = jnp.split(desired_task_acceleration, 7)
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

    task_objective = jnp.sum(
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
        jnp.sum(left_arm_dv ** 2) 
        + jnp.sum(right_arm_dv ** 2)
    )

    task_weight = 1.0
    arm_movement_weight = 1.0
    regularization_weight = 1e-4
    regularization_objective = regularization_weight * jnp.sum(q ** 2)
    objective_value = (
        task_weight * task_objective
        + arm_movement_weight * arm_movement
        + regularization_objective
    )

    return objective_value



def initialize_optimization(
    optimization_size: tuple[int, int, int, int, int],
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    # Split Index:
    dv_indx = optimization_size[0]
    u_indx = optimization_size[1] + dv_indx
    f_indx = optimization_size[2] + u_indx
    z_indx = optimization_size[3] + f_indx
    slack_indx = optimization_size[4] + z_indx

    split_indx = (dv_indx, u_indx, f_indx, z_indx, slack_indx)

    # Isolate optimization functions:
    equality_fn = lambda q, M, C, tau_g, B, H, H_bias, J, J_bias: equality_constraints(
        q,
        M,
        C,
        tau_g,
        B,
        H,
        H_bias,
        J,
        J_bias,
        split_indx,
    )

    friction = 0.6
    inequality_fn = lambda q, z_previous: inequality_constraints(
        q,
        z_previous,
        friction,
        split_indx,
    )

    objective_fn = lambda q, ddx_desired: objective(
        q,
        ddx_desired,
        split_indx,
    )

    # Generate optimization functions:
    A_eq_fn = jax.jit(jax.jacfwd(equality_fn))
    A_ineq_fn = jax.jit(jax.jacfwd(inequality_fn))
    H_fn = jax.jit(jax.jacfwd(jax.jacrev(objective_fn)))
    f_fn = jax.jit(jax.jacfwd(objective_fn))

    # Package optimization functions:
    equality_functions = (equality_fn, A_eq_fn)
    inequality_functions = (inequality_fn, A_ineq_fn)
    objective_functions = (objective_fn, H_fn, f_fn)

    return equality_functions, inequality_functions, objective_functions


def initialize_program(
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: MathematicalProgram,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int, int],
) -> tuple[MathematicalProgram, tuple[LinearConstraint, QuadraticCost]]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, z_previous = constraint_constants
    task_jacobian, task_bias, ddx_desired = objective_constants
    dv_size, u_size, f_size, z_size, slack_size  = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + slack_size
    q = jnp.zeros((num_variables,))

    # Generate program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias)
    A_ineq = A_ineq_fn(q, z_previous)
    ub_ineq = -b_ineq_fn(q, z_previous)
    # Last 4 constraints are the zero moment constraints:
    lb_ineq = jnp.concatenate(
        [
            jnp.NINF * jnp.ones_like(ub_ineq[:-4]),
            -ub_ineq[-4:],
        ],
    )

    # General Box Constraints:
    num_box_constraints = dv_size + u_size + f_size + z_size
    A_box = jnp.eye(N=num_box_constraints, M=num_variables)
    lb_reaction_forces = jnp.array(
        [
            jnp.NINF, jnp.NINF, jnp.NINF, jnp.NINF, jnp.NINF, 0.0, 
            jnp.NINF, jnp.NINF, jnp.NINF, jnp.NINF, jnp.NINF, 0.0,
        ]
    )
    leg_torque_bounds = np.array([
        116.0, 70.0, 206.0, 220.0, 35.0, 35.0
    ])
    arm_torque_bounds = np.array([
        35.0, 35.0, 35.0, 35.0
    ])
    torque_bounds = np.concatenate(
        [leg_torque_bounds, arm_torque_bounds, leg_torque_bounds, arm_torque_bounds],
    )
    lb_torque = -torque_bounds
    lb_box = jnp.concatenate(
        [
            jnp.NINF * jnp.ones((dv_size,)),
            lb_torque,
            jnp.NINF * jnp.ones((f_size,)),
            lb_reaction_forces
        ],
    )
    ub_reaction_forces = jnp.inf * jnp.ones((z_size,))
    ub_torque = torque_bounds
    ub_box = jnp.concatenate(
        [
            jnp.inf * jnp.ones((dv_size,)),
            ub_torque,
            jnp.inf * jnp.ones((f_size,)),
            ub_reaction_forces,
        ],
    )

    H = H_fn(q, ddx_desired)
    f = f_fn(q, ddx_desired)

    # Convert to sparse:
    A = np.vstack(
        [A_eq, A_ineq, A_box],
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])

    H = np.asarray(H)
    f = np.asarray(f)
    
    # Drake MP:
    opt_vars = program.NewContinuousVariables(num_variables, "q")

    # Add Constraints:
    constraint_handle = program.AddLinearConstraint(
        A=A,
        lb=lb,
        ub=ub,
        vars=opt_vars,
    )

    # Add Objective:
    objective_handle = program.AddQuadraticCost(
        Q=H,
        b=f,
        vars=opt_vars,
    )

    return program, (constraint_handle, objective_handle)


def update_program(
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: MathematicalProgram,
    program_handles: tuple[LinearConstraint, QuadraticCost],
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int, int],
) -> tuple[Solution, MathematicalProgram, tuple[LinearConstraint, QuadraticCost]]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, z_previous = constraint_constants
    task_jacobian, task_bias, ddx_desired = objective_constants
    dv_size, u_size, f_size, z_size, slack_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + slack_size
    q = np.zeros((num_variables,))

    # Precondition Matricies:
    # matrix_conditioning = (task_jacobian, task_bias)
    # conditioned_matrix = condition_matrix(matrix_conditioning)
    # task_jacobian, task_bias = conditioned_matrix

    # Generate program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias)
    A_ineq = A_ineq_fn(q, z_previous)
    ub_ineq = -b_ineq_fn(q, z_previous)
    lb_ineq = np.concatenate(
        [
            np.NINF * np.ones_like(ub_ineq[:-4]),
            -ub_ineq[-4:],
        ],
    )

    # General Box Constraints:
    num_box_constraints = dv_size + u_size + f_size + z_size
    A_box = np.eye(N=num_box_constraints, M=num_variables)
    lb_reaction_forces = np.array(
        [
            np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, 0.0, 
            np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, 0.0,
        ]
    )
    leg_torque_bounds = np.array([
        116.0, 70.0, 206.0, 220.0, 35.0, 35.0
    ])
    arm_torque_bounds = np.array([
        35.0, 35.0, 35.0, 35.0
    ])
    torque_bounds = np.concatenate(
        [leg_torque_bounds, arm_torque_bounds, leg_torque_bounds, arm_torque_bounds],
    )
    lb_torque = -torque_bounds
    lb_box = np.concatenate(
        [
            np.NINF * np.ones((dv_size,)),
            lb_torque,
            np.NINF * np.ones((f_size,)),
            lb_reaction_forces
        ],
    )
    ub_reaction_forces = np.inf * np.ones((z_size,))
    ub_torque = torque_bounds
    ub_box = np.concatenate(
        [
            np.inf * np.ones((dv_size,)),
            ub_torque,
            np.inf * np.ones((f_size,)),
            ub_reaction_forces,
        ],
    )

    H = H_fn(q, ddx_desired)
    f = f_fn(q, ddx_desired)

    # Convert to sparse:
    A = np.vstack(
        [A_eq, A_ineq, A_box],
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])
    
    H = np.asarray(H)
    f = np.asarray(f)

    # Update program:
    linear_constraint, quadratic_cost = program_handles
    linear_constraint.evaluator().UpdateCoefficients(
        new_A=A,
        new_lb=lb,
        new_ub=ub,
    )
    quadratic_cost.evaluator().UpdateCoefficients(
        new_Q=H,
        new_b=f,
    )


    solution = Solve(program)

    return solution, program, (linear_constraint, quadratic_cost)