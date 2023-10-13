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
            body_name="left-hand_link",
            base_body_name="world",
            q=q,
            qd=qd,
        )

        left_leg_act_ids = [0, 1, 2, 3, 4, 5]
        left_arm_act_ids = [6, 7, 8, 9]
        right_leg_act_ids = [10, 11, 12, 13, 14, 15]
        right_arm_act_ids = [16, 17, 18, 19]
        left_leg_act_joint_ids = [0, 1, 2, 3, 6, 7]
        left_arm_act_joint_ids = [10, 11, 12, 13]
        right_leg_act_joint_ids  = [14, 15, 16, 17, 20, 21]
        right_arm_act_joint_ids = [24, 25, 26, 27]

        B = np.zeros((plant.num_velocities(), plant.num_actuators()))
        B[left_arm_act_joint_ids, left_arm_act_ids] = 1.0

        # OSC:
        time = context.get_time()
        a = (1/4)
        rate = a * time
        r = 0.2
        xc = 0.3
        yc = 0.3

        x = xc + r * np.cos(rate)
        y = yc + r * np.sin(rate)

        dx = -r * a * np.sin(rate)
        dy = r * a * np.cos(rate)

        ddx = -r * a * a * np.cos(rate)
        ddy = -r * a * a * np.sin(rate)

        zero_vector = np.zeros((3,))
        ddx_desired = np.array([0, 0, 0, 0, ddx, ddy])
        dx_desired = np.array([0, 0, 0, 0, dx, dy])
        x_desired = np.array([0, 0, 0, 0.5, x, y])
        kp = 500
        kd = 2 * np.sqrt(kp)
        task_position = task_space_transform.translation()
        task_velocity = (spatial_velocity_jacobian @ qd)[3:]
        x_task = np.concatenate([zero_vector, task_position])
        dx_task = np.concatenate([zero_vector, task_velocity])
        control_desired = ddx_desired + kp * (x_desired - x_task) + kd * (dx_desired - dx_task)

        prog = MathematicalProgram()
        dv = prog.NewContinuousVariables(plant.num_velocities(), "dv")
        u = prog.NewContinuousVariables(plant.num_actuators(), "u")
        prog.AddBoundingBoxConstraint(
            -100, 100, u,
        )
        dynamics = M @ dv + C - tau_g
        control = B @ u
        for i in range(plant.num_velocities()):
            prog.AddLinearConstraint(
                dynamics[i] - control[i] == 0,
            )
        ddx_task = bias_spatial_acceleration + spatial_velocity_jacobian @ dv
        prog.AddQuadraticCost(
            np.sum((ddx_task - control_desired) ** 2),
        )

        results = Solve(prog)
        u = results.GetSolution(u)

        conxtext = simulator.get_context()
        actuation_context = actuation_source.GetMyContextFromRoot(conxtext)
        actuation_vector = u
        mutable_actuation_vector = actuation_source.get_mutable_source_value(
            actuation_context,
        )
        mutable_actuation_vector.set_value(actuation_vector)

        # print(u[left_arm_act_ids])

        # Get current time and set target time:
        current_time = conxtext.get_time()
        target_time = current_time + dt


if __name__ == "__main__":
    app.run(main)
