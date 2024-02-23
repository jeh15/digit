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
    plant_context = plant.CreateDefaultContext()

    # # Set Default Position:
    # default_position = np.array(
    #     [
    #         9.99999899e-01, -4.61573022e-05, 4.74404927e-04, -1.40450514e-05,
    #         4.59931778e-02, -1.77557628e-04, 1.03043887e+00,
    #         3.65207270e-01, -7.69435176e-03, 3.15664484e-01, 3.57537366e-01,
    #         -3.30752611e-01, -1.15794714e-02, -1.31615552e-01, 1.24398172e-01,
    #         1.30620121e-01, -1.15685622e-02,
    #         -1.50543436e-01, 1.09212242e+00, 1.59629876e-04, -1.39115280e-01,
    #         -3.65746560e-01, 7.48726435e-03, -3.15664484e-01, -3.57609271e-01,
    #         3.30800563e-01, 1.16105788e-02, 1.31500503e-01, -1.24536230e-01,
    #         -1.30630449e-01, 1.11680197e-02,
    #         1.50514674e-01, -1.09207448e+00, -1.74969684e-04, 1.39105692e-01,
    #     ]
    # )

    # # Set Default State:
    # default_velocity = np.zeros((plant.num_velocities(),))
    # plant.SetPositions(
    #     context=plant_context,
    #     q=default_position,
    # )
    # plant.SetVelocities(
    #     context=plant_context,
    #     v=default_velocity,
    # )

    # Initialize Systems:
    osc_rate = 1.0 / 100.0
    update_rate = 1.0 / 1000.0
    warmup_time = 3.0
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
    )
    agility_publisher = builder.AddSystem(driver_agility_publisher)

    driver_message_handler = websocket_module.MessageHandler(
        num_messengers=2,
    )
    message_handler = builder.AddSystem(driver_message_handler)

    driver_message_publisher = websocket_module.MessagePublisher()
    message_publisher = builder.AddSystem(driver_message_publisher)

    driver_websocket = websocket_module.WebsocketModule(
        ip_address="localhost",
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

    # context = simulator.get_context()
    # plant_context = plant.GetMyContextFromRoot(context)

    # plant.SetPositions(
    #     context=plant_context,
    #     q=default_position,
    # )
    # plant.SetVelocities(
    #     context=plant_context,
    #     v=default_velocity,
    # )

    # Initialize Digit Communication before Simulator Initialization:
    digit_api.initialize_communication(
        "127.0.0.1",
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
