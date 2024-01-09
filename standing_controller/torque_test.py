import os
from absl import app

import numpy as np
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

import matplotlib.pyplot as plt

# Custom Imports:
import model_utilities
import dynamics_utilities
import digit_utilities

np.set_printoptions(precision=3)


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
    dt = 0.001
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

    # Weld base to World:
    base_frame = plant.GetFrameByName("base_link")
    plant.WeldFrames(plant.world_frame(), base_frame)

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

    # Set Simulation Parameters:
    end_time = 20.0
    current_time = 0.0

    context = simulator.get_context()

    # Initialize Time:
    target_time = context.get_time() + dt
    current_time = context.get_time()

    # Run Simulation:
    A = []
    B = []
    p = []
    r = []
    motor_A = []
    motor_B = []

    qd_prev = np.zeros(plant.num_positions(), dtype=np.float64)

    while current_time < end_time:
        # Advance simulation:
        simulator.AdvanceTo(target_time)

        # Get current context:
        context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(context)

        q = plant.GetPositions(plant_context)
        qd = plant.GetVelocities(plant_context)

        M, C, tau_g, plant, plant_context = dynamics_utilities.get_dynamics(
            plant=plant,
            context=plant_context,
            q=q,
            qd=qd,
        )

        dv = np.linalg.inv(M) @ (-C)

        # dv = (qd - qd_prev) / dt
        # qd_prev = qd

        torque = np.zeros(plant.num_actuators(), dtype=np.float64)

        if current_time >= 10.0:
            torque_A = 3.0 * np.sin(2 * np.pi * current_time)
            torque_B = 0.0 * np.sin(2 * np.pi * current_time)
            torque[digit_idx.left_toe_a["actuation_idx"]] = torque_A
            torque[digit_idx.left_toe_b["actuation_idx"]] = torque_B

            motor_A.append(torque_A)
            motor_B.append(torque_B)
            A.append(dv[digit_idx.left_toe_a["joint_idx"]])
            B.append(dv[digit_idx.left_toe_b["joint_idx"]])
            p.append(dv[8])
            r.append(dv[9])

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

    # Pitch:
    motor_A = np.array(motor_A)
    motor_B = np.array(motor_B)
    A = np.array(A)
    B = np.array(B)
    p = np.array(p)
    r = np.array(r)

    # pc = 5 * (B + A)
    pc = 1.75 * (B - A)

    rc = 5 * (B + A)


    # Plot:
    fig, ax = plt.subplots()
    # ax.plot(motor_A, label="motor_A")
    # ax.plot(motor_B, label="motor_B")
    # ax.plot(A, label="A")
    # ax.plot(B, label="B")
    ax.plot(p, label="p")
    # ax.plot(r, label="r")
    ax.plot(pc, label="fit_p")
    # ax.plot(rc, label="fit_r")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    app.run(main)
