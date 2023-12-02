import os
from absl import app

import numpy as np
from pydrake.visualization import AddDefaultVisualization
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Meshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, DiscreteContactSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

# Custom Imports:
import model_utilities
import digit_utilities


def main(argv=None):
    # Load convenience class for digit:
    digit_idx = digit_utilities.DigitUtilities(floating_base=False)

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

    # Weld Base to World:
    model_utilities.teststand_configuration(plant=plant)

    # Add Terrain:
    model_utilities.add_terrain(plant=plant)

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

    # Set Default Position:
    default_position = np.array(
        [
            3.65207270e-01, -7.69435176e-03, 3.15664484e-01, 3.57537366e-01,
            -3.30752611e-01, -1.15794714e-02, -1.31615552e-01, 1.24398172e-01,
            1.30620121e-01, -1.15685622e-02, -1.50543436e-01, 1.09212242e+00,
            1.59629876e-04, -1.39115280e-01, -3.65746560e-01, 7.48726435e-03,
            -3.15664484e-01, -3.57609271e-01, 3.30800563e-01, 1.16105788e-02,
            1.31500503e-01, -1.24536230e-01, -1.30630449e-01, 1.11680197e-02,
            1.50514674e-01, -1.09207448e+00, -1.74969684e-04, 1.39105692e-01,
        ]
    )
    default_velocity = np.zeros((plant.num_velocities(),))
    plant.SetPositions(
        context=plant_context,
        q=default_position,
    )
    plant.SetVelocities(
        context=plant_context,
        v=default_velocity,
    )

    # Simulation Time:
    end_time = 60.0
    dt = 0.001
    current_time = 0.0

    # Update Context:
    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)
    current_time = context.get_time()

    target_time = current_time + dt

    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Step by dt:
        context = simulator.get_context()
        current_time = context.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
