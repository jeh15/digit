import os
from absl import app

import numpy as np
import matplotlib.pyplot as plt
import osqp
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    Meshcat,
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
    # Adding more compliance to the constraints makes the foot stay on the ground better.
    model_utilities.apply_kinematic_constraints(plant=plant, stiffness=np.inf, damping=0)

    # Add Terrain:
    model_utilities.add_terrain(plant=plant, mu_static=0.8, mu_dynamic=0.6)

    # Add auxiliary frames:
    auxiliary_frames = model_utilities.add_auxiliary_frames(plant=plant)
    constraint_frames = [
        (auxiliary_frames["left_achilles_rod"]["spring_frame"], auxiliary_frames["left_achilles_rod"]["hip_frame"]),
        (auxiliary_frames["left_toe_a"]["roll_frame"], auxiliary_frames["left_toe_a"]["motor_frame"]),
        (auxiliary_frames["left_toe_b"]["roll_frame"], auxiliary_frames["left_toe_b"]["motor_frame"]),
        (auxiliary_frames["right_achilles_rod"]["spring_frame"], auxiliary_frames["right_achilles_rod"]["hip_frame"]),
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
            4.59931778e-02, -1.77557628e-04, 1.03043887e+00, 3.65207270e-01,
            -7.69435176e-03, 3.15664484e-01, 3.57537366e-01, -3.30752611e-01,
            -1.15794714e-02, -1.31615552e-01, 1.24398172e-01, 1.30620121e-01,
            -1.15685622e-02, -1.50543436e-01, 1.09212242e+00, 1.59629876e-04,
            -1.39115280e-01, -3.65746560e-01, 7.48726435e-03, -3.15664484e-01,
            -3.57609271e-01, 3.30800563e-01, 1.16105788e-02, 1.31500503e-01,
            -1.24536230e-01, -1.30630449e-01, 1.11680197e-02, 1.50514674e-01,
            -1.09207448e+00, -1.74969684e-04, 1.39105692e-01,
        ]
    )

    actuated_joints_idx = np.concatenate(
        list(
            digit_idx.actuated_joints_idx.values()
        ),
        axis=0,
    )
    desired_position = default_position[actuated_joints_idx]

    plant.SetDefaultPositions(
        q=default_position,
    )

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

    B = digit_idx.control_matrix

    # Set Simulation Parameters:
    end_time = 60.0
    dt = 0.001
    current_time = 0.0

    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Initialize Time:
    target_time = context.get_time() + dt
    current_time = context.get_time()

    # Run Simulation:
    i = 0
    left_constraint_length = []
    right_constraint_length = []
    left_toe_a_constraint_length = []
    left_toe_b_constraint_length = []
    right_toe_a_constraint_length = []
    right_toe_b_constraint_length = []

    fig, ax = plt.subplots()
    lr = ax.plot((),(), label="Left Achilles Rod")
    rr = ax.plot((),(), label="Right Achilles Rod")
    la = ax.plot((),(), label="Left Toe A")
    lb = ax.plot((),(), label="Left Toe B")
    ra = ax.plot((),(), label="Right Toe A")
    rb = ax.plot((),(), label="Right Toe B")
    ax.legend(loc="upper right")
    ax.set_ylim([0, 1.0])
    ax.set_xlabel("Time")
    ax.set_ylabel("Constraint Length")
    ax.set_title("Constraint Length over Time")

    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        left_achilles_rod = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["left_achilles_rod"]["spring_frame"],
            frame_B=auxiliary_frames["left_achilles_rod"]["hip_frame"],
        ).translation()

        left_constraint_length.append(
            np.linalg.norm(
                left_achilles_rod,
            )
        )

        right_achilles_rod = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["right_achilles_rod"]["spring_frame"],
            frame_B=auxiliary_frames["right_achilles_rod"]["hip_frame"],
        ).translation()

        right_constraint_length.append(
            np.linalg.norm(
                right_achilles_rod,
            )
        )

        left_toe_a = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["left_toe_a"]["roll_frame"],
            frame_B=auxiliary_frames["left_toe_a"]["motor_frame"],
        ).translation()

        left_toe_a_constraint_length.append(
            np.linalg.norm(
                left_toe_a,
            )
        )

        left_toe_b = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["left_toe_b"]["roll_frame"],
            frame_B=auxiliary_frames["left_toe_b"]["motor_frame"],
        ).translation()

        left_toe_b_constraint_length.append(
            np.linalg.norm(
                left_toe_b,
            )
        )

        right_toe_a = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["right_toe_a"]["roll_frame"],
            frame_B=auxiliary_frames["right_toe_a"]["motor_frame"],
        ).translation()

        right_toe_a_constraint_length.append(
            np.linalg.norm(
                left_toe_a,
            )
        )

        right_toe_b = plant.CalcRelativeTransform(
            context=plant_context,
            frame_A=auxiliary_frames["right_toe_b"]["roll_frame"],
            frame_B=auxiliary_frames["right_toe_b"]["motor_frame"],
        ).translation()

        right_toe_b_constraint_length.append(
            np.linalg.norm(
                left_toe_b,
            )
        )

        if i % 10 == 0:
            lr[0].set_data(
                np.arange(len(left_constraint_length)),
                np.asarray(left_constraint_length),
            )
            rr[0].set_data(
                np.arange(len(right_constraint_length)),
                np.asarray(right_constraint_length),
            )
            la[0].set_data(
                np.arange(len(left_toe_a_constraint_length)),
                np.asarray(left_toe_a_constraint_length),
            )
            lb[0].set_data(
                np.arange(len(left_toe_b_constraint_length)),
                np.asarray(left_toe_b_constraint_length),
            )
            ra[0].set_data(
                np.arange(len(right_toe_a_constraint_length)),
                np.asarray(right_toe_a_constraint_length),
            )
            rb[0].set_data(
                np.arange(len(right_toe_b_constraint_length)),
                np.asarray(right_toe_b_constraint_length),
            )
            ax.relim()
            ax.autoscale_view()
            plt.show()

        # PD Control Law:
        kp = 200.0
        error = default_position[7:] - q[7:]
        control_input = kp * error
        torque = B[6:, :].T @ control_input

        # Unpack Optimization Solution:
        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = torque
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt
        i += 1


if __name__ == "__main__":
    app.run(main)
