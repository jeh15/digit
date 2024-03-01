import os
from absl import app

import numpy as np
from pydrake.geometry import (
    MeshcatVisualizer,
    Meshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    DiscreteContactSolver,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# Utility Imports:
import model_utilities
import digit_utilities

# LeafSystem Imports:
import controller_module
import taskspace_module
import trajectory_module
import agility_module
import websocket_module
import safety_module

# Digit API:
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
    meshcat = Meshcat(port=7005)

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
    model_utilities.apply_kinematic_constraints(plant=plant, stiffness=np.inf, damping=0.0)

    # Add Reflected Inertia:
    model_utilities.add_reflected_inertia(plant=plant)

    # Add Terrain:
    model_utilities.add_terrain(plant=plant, mu_static=0.8, mu_dynamic=0.6)

    # Add auxiliary frames:
    auxiliary_frames = model_utilities.add_auxiliary_frames(plant=plant)
    constraint_frames = [
        (auxiliary_frames["left_toe_a"]["roll_frame"], auxiliary_frames["left_toe_a"]["motor_frame"]),
        (auxiliary_frames["left_toe_b"]["roll_frame"], auxiliary_frames["left_toe_b"]["motor_frame"]),
        (auxiliary_frames["right_toe_a"]["roll_frame"], auxiliary_frames["right_toe_a"]["motor_frame"]),
        (auxiliary_frames["right_toe_b"]["roll_frame"], auxiliary_frames["right_toe_b"]["motor_frame"]),
    ]

    # Finalize:
    plant.Finalize()

    # Initialize Systems:
    osc_rate = 1.0 / 100.0
    update_rate = 1.0 / 1000.0
    warmup_time = 3.0 + osc_rate
    run_time = 30.0

    driver_osc_controller = controller_module.OSC(
        plant=plant,
        digit_idx=digit_idx,
        constraint_frames=constraint_frames,
        update_rate=osc_rate,
    )
    osc_controller = builder.AddSystem(driver_osc_controller)

    driver_pid_controller = controller_module.PID(
        plant=plant,
        digit_idx=digit_idx,
        update_rate=update_rate,
    )
    pid_controller = builder.AddSystem(driver_pid_controller)

    driver_taskspace_projection = taskspace_module.TaskSpace(
        plant=plant,
        update_rate=update_rate,
    )
    taskspace_projection = builder.AddSystem(driver_taskspace_projection)

    driver_trajectory_system = trajectory_module.TrajectorySystem(
        plant=plant,
        update_rate=update_rate,
        warmup_time=warmup_time,
    )
    trajectory_system = builder.AddSystem(driver_trajectory_system)

    driver_context_system = agility_module.AgilityContextSystem(
        plant=plant,
        digit_idx=digit_idx,
    )
    context_system = builder.AddSystem(driver_context_system)

    driver_agility_publisher = agility_module.AgilityPublisher(
        plant=plant,
        digit_idx=digit_idx,
        update_rate=update_rate,
        warmup_time=warmup_time,
    )
    agility_publisher = builder.AddSystem(driver_agility_publisher)

    driver_message_handler = websocket_module.MessageHandler(
        num_messengers=2,
    )
    message_handler = builder.AddSystem(driver_message_handler)

    driver_message_publisher = websocket_module.MessagePublisher()
    message_publisher = builder.AddSystem(driver_message_publisher)

    # driver_websocket = websocket_module.WebsocketModule(
    #     ip_address="localhost",
    #     port=8080,
    #     update_rate=update_rate,
    # )
    driver_websocket = websocket_module.WebsocketModule(
        ip_address="10.10.1.1",
        port=8080,
        update_rate=update_rate,
    )
    websocket = builder.AddSystem(driver_websocket)

    driver_safety_controller = safety_module.SafetyController(
        plant=plant,
        digit_idx=digit_idx,
        update_rate=update_rate,
    )
    safety_controller = builder.AddSystem(driver_safety_controller)

    # Context System -> OSC Controller:
    builder.Connect(
        context_system.get_output_port(driver_context_system.plant_context_port),
        osc_controller.get_input_port(driver_osc_controller.plant_context_port),
    )

    # Context System -> PID Controller:
    builder.Connect(
        context_system.get_output_port(driver_context_system.plant_context_port),
        pid_controller.get_input_port(driver_pid_controller.plant_context_port),
    )

    # Context System -> Task Space Projection:
    builder.Connect(
        context_system.get_output_port(driver_context_system.plant_context_port),
        taskspace_projection.get_input_port(driver_taskspace_projection.plant_context_port),
    )

    # Context System -> Safety Controller:
    builder.Connect(
        context_system.get_output_port(driver_context_system.plant_context_port),
        safety_controller.get_input_port(driver_safety_controller.plant_context_port),
    )

    # OSC Controller -> Safety Controller:
    builder.Connect(
        osc_controller.get_output_port(driver_osc_controller.solution_port),
        safety_controller.get_input_port(driver_safety_controller.solution_port),
    )

    # PID Controller -> OSC Controller:
    builder.Connect(
        pid_controller.get_output_port(driver_pid_controller.control_port),
        osc_controller.get_input_port(driver_osc_controller.desired_ddx_port),
    )

    # Trajectory System -> PID Controller:
    builder.Connect(
        trajectory_system.get_output_port(driver_trajectory_system.trajectory_port),
        pid_controller.get_input_port(driver_pid_controller.trajectory_port),
    )

    # Task Space Projection -> PID Controller:
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_jacobian_port),
        pid_controller.get_input_port(driver_pid_controller.task_jacobian_port),
    )
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_transform_rotation_port),
        pid_controller.get_input_port(driver_pid_controller.task_transform_rotation_port),
    )
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_transform_translation_port),
        pid_controller.get_input_port(driver_pid_controller.task_transform_translation_port),
    )

    # Task Space Projection -> Trajectory Module:
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_transform_rotation_port),
        trajectory_system.get_input_port(driver_trajectory_system.task_transform_rotation_port),
    )
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_transform_translation_port),
        trajectory_system.get_input_port(driver_trajectory_system.task_transform_translation_port),
    )

    # Task Space Projection -> OSC Controller:
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_jacobian_port),
        osc_controller.get_input_port(driver_osc_controller.task_jacobian_port),
    )
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_bias_port),
        osc_controller.get_input_port(driver_osc_controller.task_bias_port),
    )

    # Message Handler -> Websocket:
    builder.Connect(
        message_handler.get_output_port(driver_message_handler.message_port),
        websocket.get_input_port(driver_websocket.message_port),
    )

    # Safety Controller -> Message Handler:
    builder.Connect(
        safety_controller.get_output_port(driver_safety_controller.message_port),
        message_handler.get_input_port(driver_message_handler.input_ports[0]),
    )

    # Message Publisher -> Message Handler:
    builder.Connect(
        message_publisher.get_output_port(driver_message_publisher.message_port),
        message_handler.get_input_port(driver_message_handler.input_ports[1]),
    )

    # Safety Controller -> Agility Publisher:
    builder.Connect(
        safety_controller.get_output_port(driver_safety_controller.command_port),
        agility_publisher.get_input_port(driver_agility_publisher.command_port),
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

    # Initialize Digit Communication before Simulator Initialization:
    # digit_api.initialize_communication(
    #     "127.0.0.1",
    #     25501,
    #     25500,
    # )

    digit_api.initialize_communication(
        "10.10.1.1",
        25501,
        25500,
    )

    digit_api.wait_for_connection()

    simulator.Initialize()

    # Soft start:
    simulator.AdvanceTo(warmup_time)

    # Send out message to initialize low-level-api:
    context = simulator.get_context()
    system_context = message_publisher.GetMyContextFromRoot(context)
    message_publisher.ForcedPublish(system_context)

    # low-level-api Control:
    target_time = warmup_time + run_time
    simulator.AdvanceTo(target_time)


if __name__ == "__main__":
    app.run(main)
