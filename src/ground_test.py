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
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import RollPitchYaw, RigidTransform

# Custom Imports:
import dynamics_utilities
import model_utilities


def main(argv=None):
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

    # Move Base Above Ground:
    base_frame = plant.GetFrameByName("base_link")
    base_link_height = 0.9
    base_transform = RigidTransform(
        p=np.asarray([0, 0, base_link_height]),
    )
    plant.SetDefaultFreeBodyPose(
        body=base_frame.body(),
        X_WB=base_transform,
    )

    # Apply closed loop kinematic constraints:
    plant = model_utilities.apply_kinematic_constraints(plant=plant)

    # Add ground:
    plant = model_utilities.add_terrian(plant=plant)

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

    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()

        # Get current time and set target time:
        current_time = context.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
