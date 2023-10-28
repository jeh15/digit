import os
from absl import app

import numpy as np
import osqp
from pydrake.geometry import (
    Meshcat,
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

    # Spatial Representation:
    dv_size, u_size = plant.num_velocities(), plant.num_actuators()

    # Initialize Digit Communication:
    digit_api.initialize_communication(
        "127.0.0.1",
        25501,
        25500,
    )

    digit_api.wait_for_connection()

    current_time = dt
    target_time = current_time

    # Run Simulation:
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        # Calculate Drake:
        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        foot_position_drake = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=plant.world_frame(),
            frame_B=plant.GetFrameByName("right-foot_link"),
        )

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

        foot_position_mujoco = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=plant.world_frame(),
            frame_B=plant.GetFrameByName("right-foot_link"),
        )

        # Set positions and velocities: back to drake
        # plant.SetPositions(plant_context, q)
        # plant.SetVelocities(plant_context, qd)

        # Calculate Desired Control: Drake
        kp = 200
        kd = 0.0

        x_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_task = q[digit_idx.actuated_joints_idx["right_leg"]]
        dx_task = qd[digit_idx.actuated_joints_idx["right_leg"]]

        control_desired = np.zeros((u_size,))
        control_desired[digit_idx.actuation_idx["right_leg"]] = kp * (x_desired - x_task) - kd * (dx_task)

        # Calculate Desired Control: Mujoco
        x_desired_mujoco = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_task_mujoco = q_mujoco[digit_idx.actuated_joints_idx["right_leg"]]
        dx_task_mujoco = qd_mujoco[digit_idx.actuated_joints_idx["right_leg"]]

        control_desired_mujoco = np.zeros((u_size,))
        control_desired_mujoco[digit_idx.actuation_idx["right_leg"]] = kp * (x_desired_mujoco - x_task_mujoco) - kd * (dx_task_mujoco)

        # Send command: Drake
        actuation_context = actuation_source.GetMyContextFromRoot(context)
        actuation_vector = control_desired
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Send command: Mujoco
        torque_command = digit_idx.actuation_map(control_desired_mujoco)
        velocity_command = np.zeros((u_size,))
        damping_command = 0.75 * np.ones((u_size,))
        command = np.array([torque_command, velocity_command, damping_command]).T
        digit_api.send_command(command, 0, True)

        # Get current time and set target time:
        current_time = context.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
