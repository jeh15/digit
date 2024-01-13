from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

# Type Annotations:
from pydrake.multibody.plant import MultibodyPlant
from osqp import OSQP
Solution = Any

jax.config.update("jax_enable_x64", True)

# Surpress warnings:
# ruff: noqa: E731
# flake8: noqa: E731


@partial(jax.jit, static_argnames=['split_indx'])
def equality_constraints(
    q: jax.typing.ArrayLike,
    M: jax.typing.ArrayLike,
    C: jax.typing.ArrayLike,
    tau_g: jax.typing.ArrayLike,
    B: jax.typing.ArrayLike,
    H_: jax.typing.ArrayLike,
    H_bias: jax.typing.ArrayLike,
    J: jax.typing.ArrayLike,
    J_bias: jax.typing.ArrayLike,
    z_previous: jax.typing.ArrayLike,
    split_indx: tuple[int, int, int, int],
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
    dv_indx, u_indx, f_indx, z_indx, g_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    g = q[z_indx:g_indx]
    r = q[g_indx:]

    # Split the Task Space Jacobians:
    base_J, left_foot_J, right_foot_J = jnp.split(J, 3)
    foot_J = jnp.concatenate([left_foot_J, right_foot_J], axis=0)

    # The issue is the Achilles Rod Constraint:
    H = jnp.array([
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.
        ],
       [
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.
        ]
    ])
    H = jnp.concatenate([H, H_[2:, :]], axis=0)
    H_bias = jnp.array([0., 0., H_bias[2], H_bias[3], H_bias[4], H_bias[5]])

    # Dynamic Constraints:
    dynamics = M @ dv + C - tau_g - B @ u - H.T @ f - foot_J.T @ z

    # Kinematic Constraints:
    kinematic = H @ dv + H_bias

    # Task Accelerations:
    task_accelerations = g - (J @ dv + J_bias)

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))
    # r = jnp.reshape(r, (2, 2))

    # Zero Moment Constraint:
    # tau_x = r_y x f_z
    # tau_y = r_x x f_z
    # r = x_L, x_R, y_L, y_R
    # z[:, 0] = tau_x_L, tau_x_R
    # z[:, 1] = tau_y_L, tau_y_R
    # Original:
    # zero_moment_x = r[:, 0] - z[:, 0] / z_previous[0]
    # zero_moment_y = r[:, 1] - z[:, 1] / z_previous[1]

    # New:
    zero_moment_left_x = r[1] - z[0, 0] / z_previous[0]
    zero_moment_left_y = r[0] - z[0, 1] / z_previous[0]
    zero_moment_right_x = r[3] - z[1, 0] / z_previous[1]
    zero_moment_right_y = r[2] - z[1, 1] / z_previous[1]
    zero_moment = jnp.array([zero_moment_left_x, zero_moment_left_y, zero_moment_right_x, zero_moment_right_y])

    equality_constraints = jnp.concatenate(
        [
            dynamics,
            kinematic,
            task_accelerations,
            zero_moment,
        ],
    )

    return equality_constraints


@partial(jax.jit, static_argnames=['friction', 'split_indx'])
def inequality_constraints(
    q: jax.typing.ArrayLike,
    friction: float,
    split_indx: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    """Inequality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.

    Returns:
        The inequality constraints.

    """
    # Calculate inequality constraints:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, g_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    g = q[z_indx:g_indx]
    r = q[g_indx:]

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))
    # r = jnp.reshape(r, (2, 2))

    # Constraint: |f_x| + |f_y| <= mu * f_z
    constraint_1 = z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_2 = -z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_3 = z[:, -3] - z[:, -2] - friction * z[:, -1]
    constraint_4 = -z[:, -3] - z[:, -2] - friction * z[:, -1]

    # Torsional Friction Constraint:
    # |tau_z| <= r_y x f_x -- r_y = 0.05 -> 0.025
    # |tau_z| <= r_x x f_y -- r_x = 0.1 -> 0.05
    # |tau_z| <= r x (mu * f_z)
    r_y = 0.025
    r_x = 0.05
    constraint_5 = z[:, 2] - r_y * friction * z[:, -1]
    constraint_6 = -z[:, 2] - r_y * friction * z[:, -1]
    constraint_7 = z[:, 2] - r_x * friction * z[:, -1]
    constraint_8 = -z[:, 2] - r_x * friction * z[:, -1]

    # Constraints relatve to f_x and f_y:
    # constraint_5 = z[:, 2] - 0.1 * friction * z[:, -3]
    # constraint_6 = -z[:, 2] - 0.1 * friction * z[:, -3]
    # constraint_7 = z[:, 2] - 0.05 * friction * z[:, -2]
    # constraint_8 = -z[:, 2] - 0.05 * friction * z[:, -2]

    # Zero Moment Constraint:
    # -r_x <= r[:, 0] <= r_x
    # -r_y <= r[:, 1] <= r_y
    zero_moment_left_x = r[0] - r_x
    zero_moment_left_y = r[1] - r_y
    zero_moment_right_x = r[2] - r_x
    zero_moment_right_y = r[3] - r_y
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


@partial(jax.jit, static_argnames=['dt', 'split_indx'])
def objective(
    q: jax.typing.ArrayLike,
    desired_task_acceleration: jax.typing.ArrayLike,
    yaw_state: jax.typing.ArrayLike,
    arm_state: jax.typing.ArrayLike,
    dt: float,
    split_indx: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    # Split optimization variables:
    dv_indx, u_indx, f_indx, z_indx, g_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:z_indx]
    g = q[z_indx:g_indx]
    r = q[g_indx:]

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))
    r = jnp.reshape(r, (2, 2))

    # Split the Task Space Jacobians:
    ddx_base, ddx_left_foot, ddx_right_foot = jnp.split(g, 3)
    desired_base, desired_left_foot, desired_right_foot = jnp.split(desired_task_acceleration, 3)

    base_tracking_weight = 10.0
    foot_tracking_weight = 100.0
    base_error = base_tracking_weight * (ddx_base - desired_base) ** 2
    left_foot_error = foot_tracking_weight * (ddx_left_foot - desired_left_foot) ** 2
    right_foot_error = foot_tracking_weight * (ddx_right_foot - desired_right_foot) ** 2

    task_objective = jnp.sum(
        (
            base_error 
            + left_foot_error 
            + right_foot_error 
        ),
    )

    # Minimize Yaw Rotation:
    yaw_indx = np.array([1, 15])
    desired_heading = jnp.array([-0.00769435176, 0.00748726435])
    velocity = yaw_state[1, :] + dv[yaw_indx] * dt
    position = yaw_state[0, :] + velocity * dt
    yaw_position_objective = jnp.sum((desired_heading - position) ** 2)
    yaw_velocity_objective = jnp.sum(velocity ** 2)
    yaw_acceleration_objective = jnp.sum(dv[yaw_indx] ** 2)
    yaw_position_weight = 1.0
    yaw_velocity_weight = 0.0
    yaw_acceleration_weight = 0.0
    yaw_objective = (
        yaw_position_weight * yaw_position_objective
        + yaw_velocity_weight * yaw_velocity_objective
        + yaw_acceleration_weight * yaw_acceleration_objective
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

    # Nominal Arm Position:
    left_arm_nominal = jnp.array(
        [-1.50543436e-01, 1.09212242e+00, 1.59629876e-04, -1.39115280e-01]
    )
    right_arm_nominal = jnp.array(
        [1.50514674e-01, -1.09207448e+00, -1.74969684e-04, 1.39105692e-01]
    ) 
    left_arm_velocity = arm_state[1, :] + left_arm_dv * dt
    left_arm_position = arm_state[0, :] + left_arm_velocity * dt
    right_arm_velocity = arm_state[3, :] + right_arm_dv * dt
    right_arm_position = arm_state[2, :] + right_arm_velocity * dt
    left_arm_position_objective = jnp.sum((left_arm_nominal - left_arm_position) ** 2)
    left_arm_velocity_objective = jnp.sum(left_arm_velocity ** 2)
    right_arm_position_objective = jnp.sum((right_arm_nominal - right_arm_position) ** 2)
    right_arm_velocity_objective = jnp.sum(right_arm_velocity ** 2)
    nominal_arm_position = (
        left_arm_position_objective
        + right_arm_position_objective
    )
    arm_velocity = (
        left_arm_velocity_objective
        + right_arm_velocity_objective
    )

    # Arm Control:
    left_arm_control = jnp.sum(u[6:10] ** 2)
    right_arm_control = jnp.sum(u[16:20] ** 2)
    arm_control_objective = left_arm_control + right_arm_control

    # Regularization:
    control_objective = jnp.sum(u ** 2)
    constraint_objective = jnp.sum(f ** 2)
    x_translational_ground_reaction_objective = jnp.sum(z[:, 3] ** 2)
    y_translational_ground_reaction_objective = jnp.sum(z[:, 4] ** 2)
    z_translational_ground_reaction_objective = jnp.sum(z[:, 5] ** 2)
    x_rotational_ground_reaction_objective = jnp.sum(z[:, 0] ** 2)
    y_rotational_ground_reaction_objective = jnp.sum(z[:, 1] ** 2)
    z_rotational_ground_reaction_objective = jnp.sum(z[:, 2] ** 2)

    task_weight = 1.0
    control_weight = 0.0
    constraint_weight = 0.0
    x_translational_ground_reaction_weight = 0.00
    y_translational_ground_reaction_weight = 0.00
    z_translational_ground_reaction_weight = 0.0
    x_rotational_ground_reaction_weight = 0.0
    y_rotational_ground_reaction_weight = 0.0
    z_rotational_ground_reaction_weight = 0.0
    arm_movement_weight = 1.0
    nominal_position_weight = 1.0
    arm_velocity_weight = 0.0
    arm_control_objective_weight = 0.0
    yaw_weight = 0.0
    objective_value = (
        task_weight * task_objective 
        + control_weight * control_objective 
        + constraint_weight * constraint_objective
        + x_translational_ground_reaction_weight * x_translational_ground_reaction_objective
        + y_translational_ground_reaction_weight * y_translational_ground_reaction_objective
        + z_translational_ground_reaction_weight * z_translational_ground_reaction_objective
        + x_rotational_ground_reaction_weight * x_rotational_ground_reaction_objective
        + y_rotational_ground_reaction_weight * y_rotational_ground_reaction_objective
        + z_rotational_ground_reaction_weight * z_rotational_ground_reaction_objective
        + arm_movement_weight * arm_movement
        + nominal_position_weight * nominal_arm_position
        + arm_velocity_weight * arm_velocity
        + arm_control_objective_weight * arm_control_objective
        + yaw_weight * yaw_objective
    )

    return objective_value



def initialize_optimization(
    optimization_size: tuple[int, int, int, int, int, int],
    dt: float,
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
    g_indx = optimization_size[4] + z_indx

    split_indx = (dv_indx, u_indx, f_indx, z_indx, g_indx)

    # Isolate optimization functions:
    equality_fn = lambda q, M, C, tau_g, B, H, H_bias, J, J_bias, z_previous: equality_constraints(
        q,
        M,
        C,
        tau_g,
        B,
        H,
        H_bias,
        J,
        J_bias,
        z_previous,
        split_indx,
    )

    friction = 0.6
    inequality_fn = lambda q: inequality_constraints(
        q,
        friction,
        split_indx,
    )

    objective_fn = lambda q, ddx_desired, yaw_state, arm_state: objective(
        q,
        ddx_desired,
        yaw_state,
        arm_state,
        dt,
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
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int, int, int],
) -> OSQP:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction = constraint_constants
    ddx_desired, yaw_state, arm_state = objective_constants
    dv_size, u_size, f_size, z_size, g_size, r_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + g_size + r_size
    q = jnp.zeros((num_variables,))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction)
    A_ineq = A_ineq_fn(q)
    ub_ineq = -b_ineq_fn(q)
    # Last 4 constraints are the zero moment constraints:
    lb_ineq = jnp.concatenate(
        [jnp.NINF * jnp.ones_like(ub_ineq[:-4]), -ub_ineq[-4:]],
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

    H = H_fn(q, ddx_desired, yaw_state, arm_state)
    f = f_fn(q, ddx_desired, yaw_state, arm_state)

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq, A_box],
        )
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])
    
    H = sparse.csc_matrix(H)
    f = np.asarray(f)

    program.setup(
        P=H,
        q=f,
        A=A,
        l=lb,
        u=ub,
        verbose=True,
        warm_start=True,
        polish=True,
        rho=1e-2,
        max_iter=10000,
        eps_abs=1e-4,
        eps_rel=1e-4,
        eps_prim_inf=1e-5,
        eps_dual_inf=1e-5,
        check_termination=10,
        delta=1e-6,
        polish_refine_iter=10,
    )

    return program


def update_program(
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int, int, int],
) -> tuple[Solution, OSQP]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction = constraint_constants
    ddx_desired, yaw_state, arm_state = objective_constants
    dv_size, u_size, f_size, z_size, g_size, r_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + g_size + r_size
    q = np.zeros((num_variables,))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias, previous_ground_reaction)
    A_ineq = A_ineq_fn(q)
    ub_ineq = -b_ineq_fn(q)
    lb_ineq = np.concatenate(
        [np.NINF * np.ones_like(ub_ineq[:-4]), -ub_ineq[-4:]],
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

    H = H_fn(q, ddx_desired, yaw_state, arm_state)
    f = f_fn(q, ddx_desired, yaw_state, arm_state)

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq, A_box],
        )
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])
    
    H = sparse.csc_matrix(H)
    f = np.asarray(f)

    program.update(
        Px=sparse.triu(H).data,
        q=f,
        Ax=A.data,
        l=lb,
        u=ub,
    )

    solution = program.solve()

    return solution, program