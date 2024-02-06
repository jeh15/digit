import os
from absl import app

import casadi
import numpy as np

import time


def main(argv=None):
    dv_size = 34
    u_size = 20
    f_size = 6
    z_size = 12

    shared_library_names = [
        "equality_constraint_function.so",
        "inequality_constraint_function.so",
        "objective_function.so",
    ]

    shared_library_path = list(
        map(
            lambda shared_library_name: os.path.join(
                os.path.dirname(__file__),
                shared_library_name,
            ),
            shared_library_names,
        )
    )

    A_eq_external = casadi.external('A_eq_function', shared_library_path[0])
    b_eq_external = casadi.external('b_eq_function', shared_library_path[0])
    A_ineq_external = casadi.external('A_ineq_function', shared_library_path[1])
    b_ineq_external = casadi.external('b_ineq_function', shared_library_path[1])
    H_external = casadi.external('H_function', shared_library_path[2])
    f_external = casadi.external('f_function', shared_library_path[2])

    # Load numpy arrays:
    unittest_path = os.path.join(
        os.path.dirname(__file__),
        "unittest_files",
    )
    M = np.loadtxt(os.path.join(unittest_path, "m.txt"), delimiter=",")
    C = np.loadtxt(os.path.join(unittest_path, "c.txt"), delimiter=",")
    tau_g = np.loadtxt(os.path.join(unittest_path, "tau_g.txt"), delimiter=",")
    B = np.loadtxt(os.path.join(unittest_path, "b.txt"), delimiter=",")
    H = np.loadtxt(os.path.join(unittest_path, "h.txt"), delimiter=",")
    H_bias = np.loadtxt(
        os.path.join(unittest_path, "h_bias.txt"), delimiter=",",
    )
    J = np.loadtxt(os.path.join(unittest_path, "j.txt"), delimiter=",")
    task_bias = np.loadtxt(
        os.path.join(unittest_path, "j_bias.txt"), delimiter=",",
    )
    ground_reaction = np.loadtxt(
        os.path.join(unittest_path, "ground_reaction_forces.txt"),
        delimiter=",",
    )
    desired_task_acceleration = np.loadtxt(
        os.path.join(unittest_path, "ddx_desired.txt"), delimiter=",",
    )

    # A_eq_jax = np.loadtxt(
    #     os.path.join(unittest_path, "A_eq.txt"), delimiter=",",
    # )
    # b_eq_jax = np.loadtxt(
    #     os.path.join(unittest_path, "b_eq.txt"), delimiter=",",
    # )
    # A_ineq_jax = np.loadtxt(
    #     os.path.join(unittest_path, "A_ineq.txt"), delimiter=",",
    # )
    # b_ineq_jax = np.loadtxt(
    #     os.path.join(unittest_path, "b_ineq.txt"), delimiter=",",
    # )
    # H_jax = np.loadtxt(
    #     os.path.join(unittest_path, "hessian.txt"), delimiter=",",
    # )
    # f_jax = np.loadtxt(
    #     os.path.join(unittest_path, "gradient.txt"), delimiter=",",
    # )

    # Calculate CasADi Values:
    total_time = []
    for i in range(50):
        start_time = time.time()
        A_eq_casadi = A_eq_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            M,
            C,
            tau_g,
            B,
            H,
            H_bias,
            J,
        ).toarray()
        b_eq_casadi = b_eq_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            M,
            C,
            tau_g,
            B,
            H,
            H_bias,
            J,
        ).toarray().flatten()
        A_ineq_casadi = A_ineq_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            ground_reaction,
        ).toarray()
        b_ineq_casadi = b_ineq_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            ground_reaction,
        ).toarray().flatten()
        f_casadi = f_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            J,
            task_bias,
            desired_task_acceleration,
        ).toarray().flatten()
        H_casadi = H_external(
            np.zeros(dv_size + u_size + f_size + z_size),
            J,
            task_bias,
            desired_task_acceleration,
        ).toarray()
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time.append(elapsed_time)
        print("Elapsed Time:", elapsed_time)

    print("Average Time:", np.mean(total_time))

    # Compare:
    q = np.ones(dv_size + u_size + f_size + z_size)
    assert np.allclose(A_eq_casadi @ q, A_eq_jax @ q)
    assert np.allclose(b_eq_casadi, b_eq_jax)
    # A_ineq is failling...
    assert np.allclose(A_ineq_casadi @ q, A_ineq_jax @ q)
    assert np.allclose(b_ineq_casadi, b_ineq_jax)
    assert np.allclose(f_casadi, f_jax)
    assert np.allclose(q.T @ H_casadi @ q, q.T @ H_jax @ q)


if __name__ == "__main__":
    app.run(main)
