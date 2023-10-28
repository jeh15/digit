import os
from absl import app

import numpy as np
import osqp

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

    # Run Simulation:
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(current_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        # Get observations from Digit API:
        motor_position = digit_api.get_actuated_joint_position()
        motor_velocity = digit_api.get_actuated_joint_velocity()
        motor_torque = digit_api.get_actuated_joint_torque()
        joint_position = digit_api.get_unactuated_joint_position()
        joint_velocity = digit_api.get_unactuated_joint_velocity()

        q = digit_idx.joint_map(motor_position, joint_position)
        qd = digit_idx.joint_map(motor_velocity, joint_velocity)

        plant.SetPositions(plant_context, q)
        plant.SetVelocities(plant_context, qd)

        foot_position = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=plant.world_frame(),
            frame_B=plant.GetFrameByName("right-foot_link"),
        )

        print(foot_position.translation())

        # Calculate Desired Control:
        kp = 100
        kd = 0.0

        x_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_task = q[digit_idx.actuated_joints_idx["right_leg"]]
        dx_task = qd[digit_idx.actuated_joints_idx["right_leg"]]

        print(x_task)

        control_desired = np.zeros((u_size,))
        control_desired[digit_idx.actuation_idx["right_leg"]] = kp * (x_desired - x_task) - kd * (dx_task)

        # Send command:
        torque_command = digit_idx.actuation_map(control_desired)
        velocity_command = np.zeros((u_size,))
        damping_command = 0.75 * np.ones((u_size,))
        command = np.array([torque_command, velocity_command, damping_command]).T
        digit_api.send_command(command, 0, True)

        # Get current time and set target time:
        current_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
