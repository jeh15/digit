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

    # Setup Optimization:
    q = plant.GetPositions(plant_context)
    qd = plant.GetVelocities(plant_context)

    # Spatial Representation:
    dv_size, u_size = plant.num_velocities(), plant.num_actuators()

    current_time = dt
    target_time = current_time

    # Run Simulation:
    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        # Get observations from Digit API:
        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        # Calculate Desired Control:
        kp = 100
        # kd = 2 * np.sqrt(kp)
        kd = 0.0

        x_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_task = q[digit_idx.actuated_joints_idx["right_leg"]]
        dx_task = qd[digit_idx.actuated_joints_idx["right_leg"]]

        control_desired = np.zeros((u_size,))
        control_desired[digit_idx.actuation_idx["right_leg"]] = kp * (x_desired - x_task) - kd * (dx_task)

        # Unpack Optimization Solution:
        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = control_desired
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
