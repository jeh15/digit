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
    split_indx: tuple[int, int, int],
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
    dv_indx, u_indx, f_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:]

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
    # Dynamic Constraints:
    # Add Toe Diagonal Inertia:
    # toe_diag = jnp.eye(M.shape[0])
    # toe_diag = toe_diag.at[14, 14].set(0.1)
    # toe_diag = toe_diag.at[15, 15].set(0.1)
    # toe_diag = toe_diag.at[28, 28].set(0.1)
    # toe_diag = toe_diag.at[29, 29].set(0.1)
    # M_ = M + toe_diag
    dynamics = M @ dv + C - tau_g - B @ u - H.T @ f - J.T @ z

    # Kinematic Constraints:
    kinematic = H @ dv

    equality_constraints = jnp.concatenate(
        [dynamics, kinematic],
    )

    return equality_constraints


@partial(jax.jit, static_argnames=['friction', 'split_indx'])
def inequality_constraints(
    q: jax.typing.ArrayLike,
    friction: float,
    split_indx: tuple[int, int, int],
) -> jnp.ndarray:
    """Inequality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.

    Returns:
        The inequality constraints.

    """
    # Calculate inequality constraints:
    dv_indx, u_indx, f_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:]

    # Reshape for easier indexing:
    z = jnp.reshape(z, (2, 6))

    # Constraint: |f_x| + |f_y| <= mu * f_z
    constraint_1 = z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_2 = -z[:, -3] + z[:, -2] - friction * z[:, -1]
    constraint_3 = z[:, -3] - z[:, -2] - friction * z[:, -1]
    constraint_4 = -z[:, -3] - z[:, -2] - friction * z[:, -1]

    inequality_constraints = jnp.concatenate(
        [
            constraint_1,
            constraint_2,
            constraint_3,
            constraint_4,
        ],
    )

    return inequality_constraints


@partial(jax.jit, static_argnames=['dt', 'split_indx'])
def objective(
    q: jax.typing.ArrayLike,
    spatial_velocity_jacobian: jax.typing.ArrayLike,
    bias_spatial_acceleration: jax.typing.ArrayLike,
    desired_task_acceleration: jax.typing.ArrayLike,
    yaw_state: jax.typing.ArrayLike,
    dt: float,
    split_indx: tuple[int, int, int],
) -> jnp.ndarray:
    # Split optimization variables:
    dv_indx, u_indx, f_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:f_indx]
    z = q[f_indx:]

    z = jnp.reshape(z, (2, 6))

    # Calculate objective:
    ddx_task = bias_spatial_acceleration + spatial_velocity_jacobian @ dv

    # Quick Fix find a more principled approach:
    ddx_base, ddx_left_foot, ddx_right_foot = jnp.split(ddx_task, 3)
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
    desired_heading = jnp.array([0.36, -0.36])
    velocity = yaw_state[:, 1] + dv[yaw_indx] * dt
    position = yaw_state[:, 0] + velocity * dt
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
    arm_movement = (
        jnp.sum(dv[16:20] ** 2) 
        + jnp.sum(dv[30:34] ** 2)
    )

    # Regularization:
    control_objective = jnp.sum(u ** 2)
    constraint_objective = jnp.sum(f ** 2)
    translational_ground_reaction_objective = jnp.sum(z[:, 3:] ** 2)
    x_rotational_ground_reaction_objective = jnp.sum(z[:, 0] ** 2)
    y_rotational_ground_reaction_objective = jnp.sum(z[:, 1] ** 2)
    z_rotational_ground_reaction_objective = jnp.sum(z[:, 2] ** 2)

    task_weight = 1.0
    control_weight = 0.0
    constraint_weight = 0.01
    translational_ground_reaction_weight = 0.01
    x_rotational_ground_reaction_weight = 0.1
    y_rotational_ground_reaction_weight = 0.1
    z_rotational_ground_reaction_weight = 0.1
    arm_movement_weight = 1.0
    yaw_weight = 1.0
    objective_value = (
        task_weight * task_objective 
        + control_weight * control_objective 
        + constraint_weight * constraint_objective
        + translational_ground_reaction_weight * translational_ground_reaction_objective
        + x_rotational_ground_reaction_weight * x_rotational_ground_reaction_objective
        + y_rotational_ground_reaction_weight * y_rotational_ground_reaction_objective
        + z_rotational_ground_reaction_weight * z_rotational_ground_reaction_objective
        + arm_movement_weight * arm_movement
        + yaw_weight * yaw_objective
    )

    return objective_value



def initialize_optimization(
    optimization_size: tuple[int, int, int, int],
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
    split_indx = (dv_indx, u_indx, f_indx)

    # Isolate optimization functions:
    equality_fn = lambda q, M, C, tau_g, B, H, H_bias, J: equality_constraints(
        q,
        M,
        C,
        tau_g,
        B,
        H,
        H_bias,
        J,
        split_indx,
    )

    friction = 0.6
    inequality_fn = lambda q: inequality_constraints(
        q,
        friction,
        split_indx,
    )

    objective_fn = lambda q, J, bias, ddx_desired, yaw_state: objective(
        q,
        J,
        bias,
        ddx_desired,
        yaw_state,
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
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int],
) -> OSQP:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state = objective_constants
    dv_size, u_size, f_size, z_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size
    q = jnp.zeros((num_variables,))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian)
    A_ineq = A_ineq_fn(q)
    ub_ineq = -b_ineq_fn(q)
    lb_ineq = jnp.NINF * jnp.ones_like(ub_ineq)
    A_box = jnp.eye(num_variables)

    # General Box Constraints:
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

    H = H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state)
    f = f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state)

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
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int, int],
) -> tuple[Solution, OSQP]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state = objective_constants
    dv_size, u_size, f_size, z_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size
    q = jnp.zeros((num_variables,))

    # Generate Program Matricies:
    A_eq = np.asarray(A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian))
    b_eq = np.asarray(-b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, feet_velocity_jacobian))

    A_ineq = A_ineq_fn(q)
    ub_ineq = -b_ineq_fn(q)
    lb_ineq = jnp.NINF * jnp.ones_like(ub_ineq)
    A_box = jnp.eye(num_variables)
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
    
    H = np.asarray(H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state))
    f = np.asarray(f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired, yaw_state))

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq, A_box],
        )
    )
    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])

    H = sparse.csc_matrix(H)

    program.update(
        Px=sparse.triu(H).data,
        q=f,
        Ax=A.data,
        l=lb,
        u=ub,
    )

    solution = program.solve()

    return solution, program