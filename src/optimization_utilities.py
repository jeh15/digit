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
    H: jax.typing.ArrayLike,
    H_bias: jax.typing.ArrayLike,
    split_indx: (int, int),
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
        split_indx: The index at which the optimization
            variables are split in dv and u.

    Returns:
        The equality constraints.

        Dynamics:
        M @ dv + C @ v - tau_g - B @ u - H.T @ f = 0

        Holonomic Constraints:
        H @ dv + H_bias = 0
    """
    # Split optimization variables:
    dv_indx, u_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:]

    # Calculate equality constraints:
    dynamics = M @ dv + C - tau_g - B @ u - H.T @ f
    # dynamics = M @ dv + C - tau_g - B @ u

    kinematic = H @ dv + H_bias
    # kinematic = H @ dv

    # equality_constraints = jnp.concatenate(
    #     [dynamics, kinematic],
    # )

    equality_constraints = jnp.concatenate(
        [dynamics],
    )

    return equality_constraints


@jax.jit
def inequality_constraints(
    q: jax.typing.ArrayLike,
) -> jnp.ndarray:
    """Inequality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.

    Returns:
        The inequality constraints.

        q <= 0
    """
    # Calculate inequality constraints:
    inequality_constraints = q

    return inequality_constraints


@partial(jax.jit, static_argnames=['split_indx'])
def objective(
    q: jax.typing.ArrayLike,
    spatial_velocity_jacobian: jax.typing.ArrayLike,
    bias_spatial_acceleration: jax.typing.ArrayLike,
    desired_task_acceleration: jax.typing.ArrayLike,
    split_indx: (int, int),
) -> jnp.ndarray:
    # Split optimization variables:
    dv_indx, u_indx = split_indx
    dv = q[:dv_indx]
    u = q[dv_indx:u_indx]
    f = q[u_indx:]

    # Calculate objective:
    ddx_task = bias_spatial_acceleration + spatial_velocity_jacobian @ dv

    objective = jnp.sum((ddx_task - desired_task_acceleration) ** 2)

    return objective


def initialize_optimization(
    plant: MultibodyPlant,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    # Split Index:
    dv_indx = plant.num_velocities()
    u_indx = plant.num_actuators() + dv_indx
    split_indx = (dv_indx, u_indx)

    # Isolate optimization functions:
    equality_fn = lambda q, M, C, tau_g, B, H, H_bias: equality_constraints(
        q,
        M,
        C,
        tau_g,
        B,
        H,
        H_bias,
        split_indx,
    )

    inequality_fn = lambda q: inequality_constraints(
        q,
    )

    objective_fn = lambda q, J, bias, ddx_desired: objective(
        q,
        J,
        bias,
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
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int],
) -> OSQP:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired = objective_constants
    dv_size, u_size, f_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    q = jnp.zeros((dv_size + u_size + f_size,))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias)
    b_eq = -b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias)
    A_ineq = A_ineq_fn(q)
    lb = jnp.concatenate(
        [
            jnp.NINF * jnp.ones((dv_size,)),
            -10 * jnp.ones((u_size,)),
            jnp.NINF * jnp.ones((f_size,)),
        ],
    )
    ub = jnp.concatenate(
        [
            jnp.inf * jnp.ones((dv_size,)),
            10 * jnp.ones((u_size,)),
            jnp.inf * jnp.ones((f_size,)),
        ],
    )
    H = H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)
    f = f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq],
        )
    )
    lb = np.concatenate([b_eq, lb])
    ub = np.concatenate([b_eq, ub])
    H = sparse.csc_matrix(H)
    f = np.asarray(f)

    program.setup(
        P=H,
        q=f,
        A=A,
        l=lb,
        u=ub,
        verbose=False,
        warm_start=True,
        polish=True,
        rho=1e-2,
        max_iter=4000,
        eps_abs=1e-4,
        eps_rel=1e-4,
        eps_prim_inf=1e-6,
        eps_dual_inf=1e-6,
        check_termination=10,
        delta=1e-6,
        polish_refine_iter=5,
    )

    return program


def update_program(
    constraint_constants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    objective_constants: tuple[np.ndarray, np.ndarray, np.ndarray],
    program: OSQP,
    equality_functions: Callable,
    inequality_functions: Callable,
    objective_functions: Callable,
    optimization_size: tuple[int, int, int],
) -> tuple[Solution, OSQP]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired = objective_constants
    dv_size, u_size, f_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    q = np.zeros((dv_size + u_size + f_size,))

    # Generate Program Matricies:
    A_eq = np.asarray(A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias))
    b_eq = np.asarray(-b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias))
    A_ineq = np.asarray(A_ineq_fn(q))
    lb = jnp.concatenate(
        [
            jnp.NINF * jnp.ones((dv_size,)),
            -10 * jnp.ones((u_size,)),
            jnp.NINF * jnp.ones((f_size,)),
        ],
    )
    ub = jnp.concatenate(
        [
            jnp.inf * jnp.ones((dv_size,)),
            10 * jnp.ones((u_size,)),
            jnp.inf * jnp.ones((f_size,)),
        ],
    )
    H = np.asarray(H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired))
    f = np.asarray(f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired))

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq],
        )
    )
    lb = np.concatenate([b_eq, lb])
    ub = np.concatenate([b_eq, ub])
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