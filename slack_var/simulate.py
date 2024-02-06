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
from pydrake.multibody.meshcat import ContactVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import (
    AddFrameTriadIllustration,
)

# Custom Imports:
import model_utilities
import digit_utilities

import controller_module
import taskspace_module
import trajectory_module
import context_utilities


def main(argv=None):
    # Load convenience class for digit:
    digit_idx = digit_utilities.DigitUtilities(floating_base=True)

    # Load URDF file:
    urdf_path = "models/digit_contact.urdf"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        urdf_path,
    )

    # Start meshcat server:
    meshcat = Meshcat(port=7006)

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

    # Set Default Position:
    default_position = np.array(
        [
            9.99999899e-01, -4.61573022e-05, 4.74404927e-04, -1.40450514e-05,
            4.59931778e-02, -1.77557628e-04, 1.03043887e+00,
            3.65207270e-01, -7.69435176e-03, 3.15664484e-01, 3.57537366e-01,
            -3.30752611e-01, -1.15794714e-02, -1.31615552e-01, 1.24398172e-01,
            1.30620121e-01, -1.15685622e-02,
            -1.50543436e-01, 1.09212242e+00, 1.59629876e-04, -1.39115280e-01,
            -3.65746560e-01, 7.48726435e-03, -3.15664484e-01, -3.57609271e-01,
            3.30800563e-01, 1.16105788e-02, 1.31500503e-01, -1.24536230e-01,
            -1.30630449e-01, 1.11680197e-02,
            1.50514674e-01, -1.09207448e+00, -1.74969684e-04, 1.39105692e-01,
        ]
    )

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

    # Initialize Systems:
    driver_osc_controller = controller_module.OSC(
        plant=plant,
        digit_idx=digit_idx,
        constraint_frames=constraint_frames,
    )
    osc_controller = builder.AddSystem(driver_osc_controller)

    driver_pid_controller = controller_module.PID(
        plant=plant,
        digit_idx=digit_idx,
    )
    pid_controller = builder.AddSystem(driver_pid_controller)

    driver_taskspace_projection = taskspace_module.TaskSpace(
        plant=plant,
    )
    taskspace_projection = builder.AddSystem(driver_taskspace_projection)

    driver_context_system = context_utilities.PlantContextSystem(
        plant=plant,
    )
    context_system = builder.AddSystem(driver_context_system)

    driver_trajectory_system = trajectory_module.TrajectorySystem(
        plant=plant,
    )
    trajectory_system = builder.AddSystem(driver_trajectory_system)

    # Connect Systems:
    # Plant -> Context System:
    builder.Connect(
        plant.get_state_output_port(),
        context_system.get_input_port(
            driver_context_system.plant_state_port.get_index(),
        ),
    )

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

    # OSC Controller -> Plant:
    builder.Connect(
        osc_controller.get_output_port(driver_osc_controller.torque_port),
        plant.get_actuation_input_port(),
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

    # Task Space Projection -> OSC Controller:
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_jacobian_port),
        osc_controller.get_input_port(driver_osc_controller.task_jacobian_port),
    )
    builder.Connect(
        taskspace_projection.get_output_port(driver_taskspace_projection.task_bias_port),
        osc_controller.get_input_port(driver_osc_controller.task_bias_port),
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

    # Contact Visualizer:
    # contact_visualizer = ContactVisualizer(
    #     meshcat,
    # )
    # contact_visualizer.AddToBuilder(
    #     builder=builder,
    #     plant=plant,
    #     meshcat=meshcat,
    # )

    # Frame Triad Visualizer:
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("base_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("left-foot_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("right-foot_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("left-hand_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("right-hand_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("left-elbow_link"),
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=scene_graph,
    #     body=plant.GetBodyByName("right-elbow_link"),
    # )

    # Build diagram:
    diagram = builder.Build()

    # Create simulator:
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)

    plant.SetPositions(
        context=plant_context,
        q=default_position,
    )
    plant.SetVelocities(
        context=plant_context,
        v=default_velocity,
    )

    simulator.Initialize()

    # Advance simulation:
    target_time = 35.0
    simulator.AdvanceTo(target_time)


if __name__ == "__main__":
    app.run(main)
