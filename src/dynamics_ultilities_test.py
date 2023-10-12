import os
from absl import app

import numpy as np
from pydrake.visualization import AddDefaultVisualization
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
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


def main(argv=None):
    # Load URDF file:
    urdf_path = "models/digit_open.urdf"
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
    base_frame = plant.GetFrameByName("base_link")
    plant.WeldFrames(plant.world_frame(), base_frame)

    # Apply closed loop kinematic constraints:
    plant = model_utilities.apply_kinematic_constraints(plant=plant)

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
    target_time = dt

    # Manually change states:
    q = np.zeros(plant.num_positions())
    qd = np.zeros(plant.num_velocities())
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Dynamics Utilities:
        # q = q + np.sin(current_time) * np.ones(plant.num_positions())
        # M, C, tau_g, plant, plant_context = dynamics_utilities.get_dynamics(
        #     plant=plant,
        #     context=plant_context,
        #     q=q,
        #     qd=qd,
        # )

        # # Get pose of digit:
        # transform, plant, plant_context = dynamics_utilities.get_transform(
        #     plant=plant,
        #     context=plant_context,
        #     body_name="right-hand_link",
        #     base_body_name="right-shoulder-roll_link",
        #     q=q,
        # )

        # task_space_jacobian = dynamics_utilities.calculate_task_space_jacobian(
        #     plant=plant,
        #     context=plant_context,
        #     body_name="right-hand_link",
        #     base_body_name="base_link",
        # )

        # # Control:
        actuation_idx = [4, 14]
        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector[actuation_idx] = 5 * np.sin(current_time) * np.ones(plant.num_actuators())
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
