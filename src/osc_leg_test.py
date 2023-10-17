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
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    SolverOptions,
    IpoptSolver,
)

# Custom Imports:
import dynamics_utilities
import model_utilities
import digit_utilities


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

    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)
        actuation_context = actuation_source.GetMyContextFromRoot(context)

        # Dynamics Utilities:
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

        u = np.zeros((plant.num_actuators(),))
        kp = 25
        kd = 2 * np.sqrt(kp)

        # Get states:
        q_right_leg = q[digit_idx.actuated_joints_idx["right_leg"]]
        qd_right_leg = qd[digit_idx.actuated_joints_idx["right_leg"]]
        q_right_arm = q[digit_idx.actuated_joints_idx["right_arm"]]
        qd_right_arm = qd[digit_idx.actuated_joints_idx["right_arm"]]

        # Right Leg Controll:
        right_leg_motor = digit_idx.actuation_idx["right_leg"]
        u[right_leg_motor] = kp * (-q_right_leg) - kd * (qd_right_leg)

        # Use specific mappings:
        right_knee_motor = digit_idx.right_knee["actuation_idx"]
        right_toe_a_motor = digit_idx.right_toe_a["actuation_idx"]
        right_toe_b_motor = digit_idx.right_toe_b["actuation_idx"]
        u[right_knee_motor] = 20 * np.sin(context.get_time())
        u[right_toe_a_motor] = 0.0
        u[right_toe_b_motor] = 0.0

        # Right Arm Controll:
        right_arm_motor = digit_idx.actuation_idx["right_arm"]
        u[right_arm_motor] = kp * (-q_right_arm) - kd * (qd_right_arm)

        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = u
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
