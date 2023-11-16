import os
from absl import app

import numpy as np
import osqp
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    StartMeshcat,
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

import digit_api


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
    meshcat = StartMeshcat()

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

    end_time = 60.0
    dt = 0.001
    current_time = 0.0

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
    dv_size, u_size, f_size = plant.num_velocities(), plant.num_actuators(), 6

    B = digit_idx.control_matrix

    ddx_desired = np.zeros((velocity_jacobian.shape[0],))

    constraint_constants = (M, C, tau_g, B, H, H_bias)

    objective_constants = (
        velocity_jacobian,
        bias_acceleration,
        ddx_desired,
    )

    # Initialize Solver:
    program = osqp.OSQP()
    equality_fn, inequality_fn, objective_fn = optimization_utilities.initialize_optimization(
        plant=plant,
    )
    program = optimization_utilities.initialize_program(
        constraint_constants=constraint_constants,
        objective_constants=objective_constants,
        program=program,
        equality_functions=equality_fn,
        inequality_functions=inequality_fn,
        objective_functions=objective_fn,
        optimization_size=(dv_size, u_size, f_size),
    )
    # Create Isolated Update Function:
    update_optimization = lambda constraint_constants, objective_constants, program: optimization_utilities.update_program(
        constraint_constants,
        objective_constants,
        program,
        equality_fn,
        inequality_fn,
        objective_fn,
        (dv_size, u_size, f_size),
    )

    # Run Simulation for a few steps:
    simulator.AdvanceTo(0.5)
    target_time = simulator.get_context().get_time() + dt

    # Initialize Digit Communication:
    digit_api.initialize_communication(
        "127.0.0.1",
        25501,
        25500,
    )

    digit_api.wait_for_connection()

    current_time = dt

    # Run Simulation:
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        # Get observations from Digit API:
        motor_position = digit_api.get_actuated_joint_position()
        motor_velocity = digit_api.get_actuated_joint_velocity()
        motor_torque = digit_api.get_actuated_joint_torque()
        joint_position = digit_api.get_unactuated_joint_position()
        joint_velocity = digit_api.get_unactuated_joint_velocity()
        base_position = np.concatenate(
            [digit_api.get_base_orientation(), digit_api.get_base_position()],
        )
        base_velocity = np.concatenate(
            [digit_api.get_base_angular_velocity(), digit_api.get_base_linear_velocity()],
        )

        q = digit_idx.joint_map(motor_position, joint_position, base_position)
        qd = digit_idx.joint_map(motor_velocity, joint_velocity, base_velocity)

        # print(f"Current Position: {q}")
        # print(f"Current Velocity: {qd}")

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

        # Debug:
        # mujoco_q = q[digit_idx.actuated_joints_idx["right_leg"]]
        # drake_q = plant.GetPositions(plant_context)[digit_idx.actuated_joints_idx["right_leg"]]
        # print(f"Mujoco: {mujoco_q} \n Drake: {drake_q}")

        # Calculate Desired Control:
        kp = 0.1
        kd = 0.1 * np.sqrt(kp)

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
        right_foot_w = np.array([1.0, 0.0, 0.0, 0.0])

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

        loop_iterables = zip(task_transform, task_jacobian, position_target, rotation_target)

        # Calculate Desired Control:
        control_input = []
        for transform, J, x_target, w_target in loop_iterables:
            task_position = transform.translation()
            task_rotation = transform.rotation().ToQuaternion()
            task_velocity = J @ qd
            target_rotation = Quaternion(w_target[2])
            # From Ickes, B. P. (1970): For control purposes the last three elements of the quaternion define the roll, pitch, and yaw rotational errors.
            rotation_error = target_rotation.multiply(task_rotation.conjugate()).xyz()
            position_control = x_target[0] + kd * (x_target[1] - task_velocity[3:]) + kp * (x_target[2] - task_position)
            rotation_control = w_target[0] + kd * (w_target[1] - task_velocity[:3]) + kp * (rotation_error)
            control_input.append(
                np.concatenate([rotation_control, position_control])
            )

        control_input = np.concatenate(control_input, axis=0)

        constraint_constants = (M, C, tau_g, B, H, H_bias)

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
        constraint_force = solution.x[dv_size + u_size:]

        # Send command:
        torque_command = digit_idx.actuation_map(torque)
        velocity_command = np.zeros((u_size,))
        damping_command = 0.75 * np.ones((u_size,))
        command = np.array([torque_command, velocity_command, damping_command]).T
        digit_api.send_command(command, 0, True)

        # Unpack Optimization Solution:
        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = np.zeros_like(torque)
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # print(torque_command)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
