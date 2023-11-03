import os
from absl import app

import numpy as np
import osqp
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
    digit_idx = digit_utilities.DigitUtilities()

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

    # Weld base to World:
    model_utilities.teststand_configuration(plant=plant)

    # Apply closed loop kinematic constraints:
    model_utilities.apply_kinematic_constraints(plant=plant)

    # Add auxiliary frames:
    auxiliary_frames = model_utilities.add_auxiliary_frames(plant=plant)
    constraint_frames = [
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

    task_space_transform, spatial_velocity_jacobian, bias_spatial_acceleration = dynamics_utilities.calculate_task_space_matricies(
        plant=plant,
        context=plant_context,
        body_name="right-foot_link",
        base_body_name="world",
        q=q,
        qd=qd,
    )

    H_spatial, H_bias_spatial = dynamics_utilities.calculate_kinematic_constraints_spatial(
        plant=plant,
        context=plant_context,
        constraint_frames=constraint_frames,
        q=q,
        qd=qd,
    )

    # Control only right leg:
    joint_indx = digit_idx.actuated_joints_idx["right_leg"]
    actuation_indx = digit_idx.actuation_idx["right_leg"]
    left_leg = digit_idx.actuated_joints_idx["left_leg"]
    left_arm = digit_idx.actuated_joints_idx["left_arm"]
    right_arm = digit_idx.actuated_joints_idx["right_arm"]

    M = M[joint_indx, :][:, joint_indx]
    C = C[joint_indx]
    tau_g = tau_g[joint_indx]

    J = spatial_velocity_jacobian[:, joint_indx]
    dJ = bias_spatial_acceleration

    H = H_spatial[:, joint_indx]
    bias = H_bias_spatial

    # Spatial Representation:
    dv_size, u_size, f_size = len(joint_indx), len(actuation_indx), 18
    B = np.eye((dv_size))

    # Spatial Representation:
    control_desired = np.zeros((6,))

    constraint_constants = (M, C, tau_g, B, H, bias)

    objective_constants = (
        J,
        dJ,
        control_desired,
    )

    # Initialize Solver:
    program = osqp.OSQP()
    equality_fn, inequality_fn, objective_fn = optimization_utilities.initialize_optimization(
        plant=plant,
        optimization_size=(dv_size, u_size, f_size),
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

        q_mujoco = digit_idx.joint_map(motor_position, joint_position)
        qd_mujoco = digit_idx.joint_map(motor_velocity, joint_velocity)

        plant.SetPositions(plant_context, q_mujoco)
        plant.SetVelocities(plant_context, qd_mujoco)

        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        M, C, tau_g, plant, plant_context = dynamics_utilities.get_dynamics(
            plant=plant,
            context=plant_context,
            q=q,
            qd=qd,
        )

        task_space_transform, spatial_velocity_jacobian, bias_spatial_acceleration = dynamics_utilities.calculate_task_space_matricies(
            plant=plant,
            context=plant_context,
            body_name="right-foot_link",
            base_body_name="world",
            q=q,
            qd=qd,
        )

        H_spatial, H_bias_spatial = dynamics_utilities.calculate_kinematic_constraints_spatial(
            plant=plant,
            context=plant_context,
            constraint_frames=constraint_frames,
            q=q,
            qd=qd,
        )

        # Tracking Trajectory:
        time = current_time
        a_1 = 1/4
        a_2 = 1/4
        rate_1 = a_1 * time
        rate_2 = a_2 * time
        r_1 = 0.2
        r_2 = 0.2
        xc = 0.0
        yc = 1.2 - 0.95

        x = xc + r_1 * np.cos(rate_1)
        y = yc + r_2 * np.sin(rate_2)

        dx = -r_1 * a_1 * np.sin(rate_1)
        dy = r_2 * a_2 * np.cos(rate_2)

        ddx = -r_1 * a_1 * a_1 * np.cos(rate_1)
        ddy = -r_2 * a_2 * a_2 * np.sin(rate_2)

        # Calculate Desired Control:
        kp = 100
        kd = 2 * np.sqrt(kp)

        # Calculate Desired Control: Spatial Representation:
        zero_vector = np.zeros((3,))
        ddx_desired = np.array([0, 0, 0, ddx, 0, ddy])
        dx_desired = np.array([0, 0, 0, dx, 0, dy])
        x_desired = np.array([0, 0, 0, x, -0.1, y])
        task_position = task_space_transform.translation()
        task_velocity = (spatial_velocity_jacobian @ qd)[3:]
        x_task = np.concatenate([zero_vector, task_position])
        dx_task = np.concatenate([zero_vector, task_velocity])

        control_desired = ddx_desired + kp * (x_desired - x_task) + kd * (dx_desired - dx_task)

        # PID Control:
        kp = 100
        kd = 2 * np.sqrt(kp)

        left_leg_control = kp * (np.zeros_like(q[left_leg]) - q[left_leg]) - kd * (qd[left_leg])
        left_arm_control = kp * (np.zeros_like(q[left_arm]) - q[left_arm]) - kd * (qd[left_arm])
        right_arm_control = kp * (np.zeros_like(q[right_arm]) - q[right_arm]) - kd * (qd[right_arm])

        M = M[joint_indx, :][:, joint_indx]
        C = C[joint_indx]
        tau_g = tau_g[joint_indx]

        J = spatial_velocity_jacobian[:, joint_indx]
        dJ = bias_spatial_acceleration

        H = H_spatial[:, joint_indx]
        bias = H_bias_spatial

        constraint_constants = (M, C, tau_g, B, H, bias)
        objective_constants = (
            J,
            dJ,
            control_desired,
        )

        # Solve Optimization:
        solution, program = update_optimization(
            constraint_constants=constraint_constants,
            objective_constants=objective_constants,
            program=program,
        )

        # Unpack Optimization Solution:
        accelerations = solution.x[:dv_size]
        torque = solution.x[dv_size:dv_size + u_size]
        constraint_force = solution.x[dv_size + u_size:]

        print(torque)

        # Pad Torque with Zeros:
        arm_zeros = np.zeros((len(digit_idx.actuation_idx["left_arm"]),))
        torque = np.concatenate(
            [left_leg_control, left_arm_control, torque, right_arm_control],
        )

        # Send command:
        torque_command = digit_idx.actuation_map(torque)
        velocity_command = np.zeros_like(torque_command)
        damping_command = 0.75 * np.ones_like(torque_command)
        command = np.array([torque_command, velocity_command, damping_command]).T
        digit_api.send_command(command, 0, True)

        # Get current time and set target time:
        current_time = context.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
