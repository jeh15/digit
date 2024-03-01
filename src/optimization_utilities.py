import os
from typing import Callable, Any

import numpy as np
import casadi

from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    QuadraticCost,
    LinearConstraint,
)

# Type Annotations:
Solution = Any
ConstraintConstants = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
ObjectiveConstants = tuple[np.ndarray, np.ndarray, np.ndarray]

# Constants:
tolerance = 1e-5
scale = 1e2
significant_bits = 8
eps = np.finfo(np.float64).eps


def link_shared_library() -> tuple[
    tuple[Callable, Callable],
    tuple[Callable, Callable],
    tuple[Callable, Callable],
]:
    # Create external casadi functions:
    autogen_path = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        "autogen",
    )
    shared_library_names = [
        "equality_constraint_function.so",
        "inequality_constraint_function.so",
        "objective_function.so",
    ]

    shared_library_path = list(
        map(
            lambda shared_library_name: os.path.join(
                autogen_path,
                shared_library_name,
            ),
            shared_library_names,
        )
    )

    A_eq_function = casadi.external('A_eq_function', shared_library_path[0])
    b_eq_function = casadi.external('b_eq_function', shared_library_path[0])
    A_ineq_function = casadi.external('A_ineq_function', shared_library_path[1])
    b_ineq_function = casadi.external('b_ineq_function', shared_library_path[1])
    H_function = casadi.external('H_function', shared_library_path[2])
    f_function = casadi.external('f_function', shared_library_path[2])

    # Package optimization functions:
    equality_functions = (A_eq_function, b_eq_function)
    inequality_functions = (A_ineq_function, b_ineq_function)
    objective_functions = (H_function, f_function)

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
    A_eq_fn, b_eq_fn = equality_functions
    A_ineq_fn, b_ineq_fn = inequality_functions
    H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + slack_size
    q = np.zeros((num_variables,))

    # Generate program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias).toarray()
    b_eq = b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias).toarray().flatten()
    A_ineq = A_ineq_fn(q, z_previous).toarray()
    ub_ineq = b_ineq_fn(q, z_previous).toarray().flatten()
    # Last 4 constraints are the zero moment constraints:
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
    ub_reaction_forces[5] = 400.0
    ub_reaction_forces[-1] = 400.0
    ub_torque = torque_bounds
    ub_box = np.concatenate(
        [
            np.inf * np.ones((dv_size,)),
            ub_torque,
            np.inf * np.ones((f_size,)),
            ub_reaction_forces,
        ],
    )

    H = H_fn(q, ddx_desired).toarray()
    f = f_fn(q, ddx_desired).toarray().flatten()

    # Convert to sparse:
    A = np.vstack(
        [A_eq, A_ineq, A_box],
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])

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
    A_eq_fn, b_eq_fn = equality_functions
    A_ineq_fn, b_ineq_fn = inequality_functions
    H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size + slack_size
    q = np.zeros((num_variables,))

    # Generate program Matricies:
    A_eq = A_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias).toarray()
    b_eq = b_eq_fn(q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian, task_bias).toarray().flatten()
    A_ineq = A_ineq_fn(q, z_previous).toarray()
    ub_ineq = b_ineq_fn(q, z_previous).toarray().flatten()
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

    H = H_fn(q, ddx_desired).toarray()
    f = f_fn(q, ddx_desired).toarray().flatten()

    # Convert to sparse:
    A = np.vstack(
        [A_eq, A_ineq, A_box],
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])

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
