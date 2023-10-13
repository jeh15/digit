from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import time

# Type Annotations:
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    SolverOptions,
    OsqpSolver,
)


@partial(jax.jit, static_argnames=['split_indx'])
def equality_constraints(
    q: jax.typing.ArrayLike,
    M: jax.typing.ArrayLike,
    C: jax.typing.ArrayLike,
    tau_g: jax.typing.ArrayLike,
    B: jax.typing.ArrayLike,
    split_indx: int,
) -> jnp.ndarray:
    """Equality constraints for the dynamics of a system.

    Args:
        q: The generalized positions.
        M: The mass matrix.
        C: The Coriolis matrix.
        tau_g: The gravity vector.
        B: The actuation matrix.
        split_indx: The index at which the optimization
            variables are split in dv and u.

    Returns:
        The equality constraints.

        M @ q + C @ q + tau_g - B @ u = 0
    """
    # Split optimization variables:
    dv = q[:split_indx]
    u = q[split_indx:]

    # Calculate equality constraints:
    equality_constraints = M @ dv + C - tau_g - B @ u

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
    split_indx: int,
) -> jnp.ndarray:
    # Split optimization variables:
    dv = q[:split_indx]
    u = q[split_indx:]

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
    split_indx = plant.num_velocities()

    # Isolate optimization functions:
    equality_fn = lambda q, M, C, tau_g, B: equality_constraints(
        q,
        M,
        C,
        tau_g,
        B,
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
    constraint_constants: tuple,
    objective_constants: tuple,
    program: MathematicalProgram,
    equality_functions: tuple[callable, callable],
    inequality_functions: tuple[callable, callable],
    objective_functions: tuple[callable, callable, callable],
    optimization_size: tuple[int, int],
):
    # Unpack optimization constants and other variables:
    M, C, tau_g, B = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired = objective_constants
    dv_size, u_size = optimization_size

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    q = jnp.zeros((program.num_vars(),))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B)
    b_eq = -b_eq_fn(q, M, C, tau_g, B)
    A_ineq = A_ineq_fn(q)
    lb = jnp.concatenate(
        [
            jnp.NINF * jnp.ones((dv_size,)),
            -10 * jnp.ones((u_size,)),
        ],
    )
    ub = jnp.concatenate(
        [
            jnp.inf * jnp.ones((dv_size,)),
            10 * jnp.ones((u_size,)),
        ],
    )
    H = H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)
    f = f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)

    # Drake Functions:
    constraint_handle = program.AddLinearEqualityConstraint(
        Aeq=A_eq,
        beq=b_eq,
        vars=program.decision_variables(),
    )
    objective_handle = program.AddQuadraticCost(
        Q=H,
        b=f,
        vars=program.decision_variables(),
        is_convex=True,
    )
    program.AddLinearConstraint(
        A=A_ineq,
        lb=lb,
        ub=ub,
        vars=program.decision_variables(),
    )

    return  constraint_handle, objective_handle, program


def update_program(
    constraint_constants: tuple,
    objective_constants: tuple,
    program: MathematicalProgram,
    solver: OsqpSolver,
    solver_options: SolverOptions,
    equality_functions: tuple[callable, callable],
    inequality_functions: tuple[callable, callable],
    objective_functions: tuple[callable, callable, callable],
    optimization_handles: tuple[callable, callable],
    optimization_size: tuple[int, int],
):
    start_time = time.time()
    # Unpack optimization constants and other variables:
    M, C, tau_g, B = constraint_constants
    spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired = objective_constants
    dv_size, u_size = optimization_size
    constraint_handle, objective_handle = optimization_handles

    # Unpack optimization functions:
    b_eq_fn, A_eq_fn = equality_functions
    b_ineq_fn, A_ineq_fn = inequality_functions
    objective_fn, H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    q = jnp.zeros((program.num_vars(),))

    # Generate Program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B)
    b_eq = -b_eq_fn(q, M, C, tau_g, B)
    A_ineq = A_ineq_fn(q)
    lb = jnp.concatenate(
        [
            jnp.NINF * jnp.ones((dv_size,)),
            -10 * jnp.ones((u_size,)),
        ],
    )
    ub = jnp.concatenate(
        [
            jnp.inf * jnp.ones((dv_size,)),
            10 * jnp.ones((u_size,)),
        ],
    )
    H = H_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)
    f = f_fn(q, spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)

    # Drake Functions:
    constraint_handle.evaluator().UpdateCoefficients(
        Aeq=A_eq,
        beq=b_eq,
    )
    constraint_handle.evaluator().RemoveTinyCoefficient(1e-5)
    objective_handle.evaluator().UpdateCoefficients(
        new_Q=H,
        new_b=f,
    )

    solution = solver.Solve(
        program,
        np.zeros((program.num_vars(),)),
        solver_options,
    )

    # print(f"Optimization Time: {time.time() - start_time}")

    return solution