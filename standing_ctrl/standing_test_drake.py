import os
from absl import app

import numpy as np
import osqp
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    Meshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, DiscreteContactSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

# Custom Imports:
import dynamics_utilities
import model_utilities
import digit_utilities
import optimization_utilities


def main(argv=None):
    # Load convenience class for digit:
    digit_idx = digit_utilities.DigitUtilities(floating_base=True)

    # Load URDF file:
    urdf_path = "models/digit.urdf"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        urdf_path,
    )

    # Start meshcat server:
    meshcat = Meshcat(port=7004)

    builder = DiagramBuilder()
    time_step = 0.0005
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder,
        time_step=time_step,
    )
    plant.set_discrete_contact_solver(
        DiscreteContactSolver.kSap,
    )
    parser = Parser(plant)
    parser.AddModels(filepath)

    # Apply closed loop kinematic constraints:
    model_utilities.apply_kinematic_constraints(plant=plant)

    # Add Terrain:
    model_utilities.add_terrain(plant=plant)

    # Add auxiliary frames:
    auxiliary_frames = model_utilities.add_auxiliary_frames(plant=plant)
    constraint_frames = [
        (auxiliary_frames["left_achilles_rod"]["spring_frame"], auxiliary_frames["left_achilles_rod"]["hip_frame"]),
        (auxiliary_frames["left_toe_a"]["roll_frame"], auxiliary_frames["left_toe_a"]["motor_frame"]),
        (auxiliary_frames["left_toe_b"]["roll_frame"], auxiliary_frames["left_toe_b"]["motor_frame"]),
        (auxiliary_frames["right_achilles_rod"]["spring_frame"], auxiliary_frames["right_achilles_rod"]["hip_frame"]),
        (auxiliary_frames["right_toe_a"]["roll_frame"], auxiliary_frames["right_toe_a"]["motor_frame"]),
        (auxiliary_frames["right_toe_b"]["roll_frame"], auxiliary_frames["right_toe_b"]["motor_frame"]),
    ]

    # Finalize:
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()

    # Set Default Position:
    default_position = np.array(
        [
            9.99999899e-01, -4.61573022e-05, 4.74404927e-04, -1.40450514e-05,
            4.59931778e-02, -1.77557628e-04, 1.03043887e+00, 3.65207270e-01,
            -7.69435176e-03, 3.15664484e-01, 3.57537366e-01, -3.30752611e-01,
            -1.15794714e-02, -1.31615552e-01, 1.24398172e-01, 1.30620121e-01,
            -1.15685622e-02, -1.50543436e-01, 1.09212242e+00, 1.59629876e-04,
            -1.39115280e-01, -3.65746560e-01, 7.48726435e-03, -3.15664484e-01,
            -3.57609271e-01, 3.30800563e-01, 1.16105788e-02, 1.31500503e-01,
            -1.24536230e-01, -1.30630449e-01, 1.11680197e-02, 1.50514674e-01,
            -1.09207448e+00, -1.74969684e-04, 1.39105692e-01,
        ]
    )

    plant.SetDefaultPositions(
        q=default_position,
    )

    # Connect Vector Source to Digit's Actuators:
    actuation_vector = np.zeros(
        plant.num_actuators(),
        dtype=np.float64,
    )
    actuation_source = builder.AddSystem(
        ConstantVectorSource(actuation_vector),
    )
    builder.Connect(
        actuation_source.get_output_port(),
        plant.get_actuation_input_port(),
    )

    # Add Meshcat Visualizer:
    meshcat_visualizer = MeshcatVisualizer(
        meshcat,
    )
    meshcat_visualizer.AddToBuilder(
        builder=builder,
        scene_graph=scene_graph,
        meshcat=meshcat,
    )

    # Build diagram:
    diagram = builder.Build()

    # Create simulator:
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Set Default State:
    default_velocity = np.zeros((plant.num_velocities(),))
    plant.SetPositions(
        context=plant_context,
        q=default_position,
    )
    plant.SetVelocities(
        context=plant_context,
        v=default_velocity,
    )

    # Setup Optimization:
    q = plant.GetPositions(plant_context)
    qd = plant.GetVelocities(plant_context)

    M, C, tau_g, plant, plant_context = dynamics_utilities.get_dynamics(
        plant=plant,
        context=plant_context,
        q=q,
        qd=qd,
    )

    task_transform, velocity_jacobian, bias_acceleration = dynamics_utilities.calculate_taskspace(
        plant=plant,
        context=plant_context,
        body_name=["base_link", "left-foot_link", "right-foot_link"],
        base_body_name="world",
        q=q,
        qd=qd,
    )

    H, H_bias = dynamics_utilities.calculate_kinematic_constraints(
        plant=plant,
        context=plant_context,
        constraint_frames=constraint_frames,
        q=q,
        qd=qd,
    )

    # Translation Representation:
    dv_size, u_size, f_size, z_size = plant.num_velocities(), plant.num_actuators(), 6, 12

    B = digit_idx.control_matrix

    ddx_desired = np.zeros((velocity_jacobian.shape[0],))

    # feet_velocity_jacobian = velocity_jacobian[6:, :]
    base_J, left_foot_J, right_foot_J = np.split(velocity_jacobian, 3)
    feet_velocity_jacobian = np.concatenate([left_foot_J, right_foot_J], axis=0)
    constraint_constants = (M, C, tau_g, B, H, H_bias, feet_velocity_jacobian)

    objective_constants = (
        velocity_jacobian,
        bias_acceleration,
        ddx_desired,
    )

    # Initialize Solver:
    program = osqp.OSQP()
    equality_fn, inequality_fn, objective_fn = optimization_utilities.initialize_optimization(
        optimization_size=(dv_size, u_size, f_size, z_size),
    )
    program = optimization_utilities.initialize_program(
        constraint_constants=constraint_constants,
        objective_constants=objective_constants,
        program=program,
        equality_functions=equality_fn,
        inequality_functions=inequality_fn,
        objective_functions=objective_fn,
        optimization_size=(dv_size, u_size, f_size, z_size),
    )
    # Create Isolated Update Function:
    update_optimization = lambda constraint_constants, objective_constants, program: optimization_utilities.update_program(
        constraint_constants,
        objective_constants,
        program,
        equality_fn,
        inequality_fn,
        objective_fn,
        (dv_size, u_size, f_size, z_size),
    )

    # Set Simulation Parameters:
    end_time = 60.0
    dt = 0.001
    current_time = 0.0

    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Initialize Time:
    target_time = context.get_time() + dt
    current_time = context.get_time()

    # Run Simulation:
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        M, C, tau_g, plant, plant_context = dynamics_utilities.get_dynamics(
            plant=plant,
            context=plant_context,
            q=q,
            qd=qd,
        )

        task_transform, velocity_jacobian, bias_acceleration = dynamics_utilities.calculate_taskspace(
            plant=plant,
            context=plant_context,
            body_name=["base_link", "left-foot_link", "right-foot_link"],
            base_body_name="world",
            q=q,
            qd=qd,
        )

        H, H_bias = dynamics_utilities.calculate_kinematic_constraints(
            plant=plant,
            context=plant_context,
            constraint_frames=constraint_frames,
            q=q,
            qd=qd,
        )

        # Calculate Desired Control:
        kp_position_base = 200.0
        kd_position_base = 1 * np.sqrt(kp_position_base)
        kp_rotation_base = 100.0
        kd_rotation_base = 2 * np.sqrt(kp_rotation_base)
        kp_position_feet = 10.0
        kd_position_feet = 2 * np.sqrt(kp_position_feet)
        kp_rotation_feet = 100.0
        kd_rotation_feet = 2 * np.sqrt(kp_rotation_feet)

        control_gains = [
            [kp_position_base, kd_position_base, kp_rotation_base, kd_rotation_base],
            [kp_position_feet, kd_position_feet, kp_rotation_feet, kd_rotation_feet],
            [kp_position_feet, kd_position_feet, kp_rotation_feet, kd_rotation_feet],
        ]

        # Base Tracking:
        # Position:
        base_ddx = np.zeros((3,))
        base_dx = np.zeros_like(base_ddx)
        base_x = np.array([0.04638328773710699, -0.00014100711268926657, 1.0308927292801415])
        # Rotation:
        base_ddw = np.zeros_like(base_ddx)
        base_dw = np.zeros_like(base_ddw)
        base_w = np.array([1.0, 0.0, 0.0, 0.0])

        # Foot Tracking:
        # Position:
        foot_ddx = np.zeros_like(base_ddx)
        foot_dx = np.zeros_like(foot_ddx)
        left_foot_x = np.array([0.009485657750110333, 0.10003118944491024, -0.0006031847782857091])
        right_foot_x = np.array([0.009501654135451067, -0.10004060651147584, -0.0006041746580776665])
        # Rotation:
        foot_ddw = np.zeros_like(base_ddx)
        foot_dw = np.zeros_like(foot_ddw)
        left_foot_w = np.array([1.0, 0.0, 0.0, 0.0])
        # left_foot_w = np.array([9.99984167e-01, 2.61407089e-04, -5.52833537e-03, 1.01688480e-03])
        right_foot_w = np.array([1.0, 0.0, 0.0, 0.0])
        # right_foot_w = np.array([9.99975894e-01, 5.17854946e-06, -6.90341587e-03, -7.45056438e-04])

        position_target = [
            [base_ddx, base_dx, base_x],
            [foot_ddx, foot_dx, left_foot_x],
            [foot_ddx, foot_dx, right_foot_x],
        ]
        rotation_target = [
            [base_ddw, base_dw, base_w],
            [foot_ddw, foot_dw, left_foot_w],
            [foot_ddw, foot_dw, right_foot_w],
        ]
        task_jacobian = np.split(velocity_jacobian, 3)

        loop_iterables = zip(task_transform, task_jacobian, position_target, rotation_target, control_gains)

        # Calculate Desired Control:
        control_input = []
        for transform, J, x_target, w_target, gains in loop_iterables:
            task_position = transform.translation()
            task_rotation = transform.rotation().ToQuaternion()
            task_velocity = J @ qd
            target_rotation = Quaternion(w_target[2])
            # From Ickes, B. P. (1970): For control purposes the last three elements of the quaternion define the roll, pitch, and yaw rotational errors.
            rotation_error = target_rotation.multiply(task_rotation.conjugate()).xyz()
            position_control = x_target[0] + gains[1] * (x_target[1] - task_velocity[3:]) + gains[0] * (x_target[2] - task_position)
            rotation_control = w_target[0] + gains[2] * (w_target[1] - task_velocity[:3]) + gains[3] * (rotation_error)
            control_input.append(
                np.concatenate([rotation_control, position_control])
            )

        control_input = np.concatenate(control_input, axis=0)

        # feet_velocity_jacobian = velocity_jacobian[6:, :]
        base_J, left_foot_J, right_foot_J = np.split(velocity_jacobian, 3)
        feet_velocity_jacobian = np.concatenate([left_foot_J, right_foot_J], axis=0)
        constraint_constants = (M, C, tau_g, B, H, H_bias, feet_velocity_jacobian)

        objective_constants = (
            velocity_jacobian,
            bias_acceleration,
            control_input,
        )

        # Solve Optimization:
        solution, program = update_optimization(
            constraint_constants=constraint_constants,
            objective_constants=objective_constants,
            program=program,
        )

        assert solution.info.status_val == 1

        # Unpack Optimization Solution:
        accelerations = solution.x[:dv_size]
        torque = solution.x[dv_size:dv_size + u_size]
        constraint_force = solution.x[dv_size + u_size:dv_size + u_size + f_size]
        reaction_force = solution.x[dv_size + u_size + f_size:]

        # Unpack Optimization Solution:
        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = torque
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
