import os
from typing import Callable, Any

import numpy as np
import casadi
from scipy import sparse

# Type Annotations:
from osqp import OSQP
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
    constraint_constants: ConstraintConstants,
    objective_constants: ObjectiveConstants,
    program: OSQP,
    equality_functions: tuple[Callable, Callable],
    inequality_functions: tuple[Callable, Callable],
    objective_functions: tuple[Callable, Callable],
    optimization_size: tuple[int, int, int, int],
) -> OSQP:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, z_previous = constraint_constants
    task_jacobian, task_bias, ddx_desired = objective_constants
    dv_size, u_size, f_size, z_size = optimization_size

    # Unpack optimization functions:
    A_eq_fn, b_eq_fn = equality_functions
    A_ineq_fn, b_ineq_fn = inequality_functions
    H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size
    q = np.zeros((num_variables,))

    # Generate Program Matricies: (b functions have already been negated)
    A_eq = A_eq_fn(
        q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian
    ).toarray()
    b_eq = b_eq_fn(
        q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian
    ).toarray().flatten()
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
    torque_bounds = np.concatenate([
        leg_torque_bounds,
        arm_torque_bounds,
        leg_torque_bounds,
        arm_torque_bounds,
    ])
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

    H = H_fn(q, task_jacobian, task_bias, ddx_desired).toarray()
    f = f_fn(q, task_jacobian, task_bias, ddx_desired).toarray().flatten()

    # Convert to sparse:
    A = sparse.csc_matrix(
        np.vstack(
            [A_eq, A_ineq, A_box],
        )
    )

    lb = np.concatenate([b_eq, lb_ineq, lb_box])
    ub = np.concatenate([b_eq, ub_ineq, ub_box])

    H = sparse.csc_matrix(H)

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
    constraint_constants: ConstraintConstants,
    objective_constants: ObjectiveConstants,
    program: OSQP,
    equality_functions: tuple[Callable, Callable],
    inequality_functions: tuple[Callable, Callable],
    objective_functions: tuple[Callable, Callable],
    optimization_size: tuple[int, int, int, int],
) -> tuple[Solution, OSQP]:
    # Unpack optimization constants and other variables:
    M, C, tau_g, B, H_constraint, H_bias, z_previous = constraint_constants
    task_jacobian, task_bias, ddx_desired = objective_constants
    dv_size, u_size, f_size, z_size = optimization_size

    # Unpack optimization functions:
    A_eq_fn, b_eq_fn = equality_functions
    A_ineq_fn, b_ineq_fn = inequality_functions
    H_fn, f_fn = objective_functions

    # Initialize optimization variables for JAX:
    num_variables = dv_size + u_size + f_size + z_size
    q = np.zeros((num_variables,))

    # Generate Program Matricies: (b functions have already been negated)
    A_eq = A_eq_fn(
        q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian
    ).toarray()
    b_eq = b_eq_fn(
        q, M, C, tau_g, B, H_constraint, H_bias, task_jacobian
    ).toarray().flatten()
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
    torque_bounds = np.concatenate([
        leg_torque_bounds,
        arm_torque_bounds,
        leg_torque_bounds,
        arm_torque_bounds,
    ])
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

    H = H_fn(q, task_jacobian, task_bias, ddx_desired).toarray()
    f = f_fn(q, task_jacobian, task_bias, ddx_desired).toarray().flatten()

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
