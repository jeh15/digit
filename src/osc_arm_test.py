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
    OsqpSolver,
)

# Custom Imports:
import dynamics_utilities
import model_utilities
import optimization_utilities


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
    dt = 0.01
    current_time = dt
    target_time = dt

    # Initial Step:
    simulator.AdvanceTo(target_time)

    # Initial Values for Optimization:
    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)

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
    left_arm_act_ids = [6, 7, 8, 9]
    left_arm_act_joint_ids = [10, 11, 12, 13]
    B_left = np.zeros((plant.num_velocities(), plant.num_actuators()))
    B_left[left_arm_act_joint_ids, left_arm_act_ids] = 1.0
    ddx_desired = np.zeros((6,))
    constraint_constants = (M, C, tau_g, B_left)
    objective_constants = (spatial_velocity_jacobian, bias_spatial_acceleration, ddx_desired)

    # Initialize Solver:
    solver = OsqpSolver()
    solver_options = SolverOptions()
    solver_options = SolverOptions()
    solver_options.SetOption(solver.solver_id(), "rho", 1e-04)
    solver_options.SetOption(solver.solver_id(), "eps_abs", 1e-06)
    solver_options.SetOption(solver.solver_id(), "eps_rel", 1e-06)
    solver_options.SetOption(solver.solver_id(), "eps_prim_inf", 1e-06)
    solver_options.SetOption(solver.solver_id(), "eps_dual_inf", 1e-06)
    solver_options.SetOption(solver.solver_id(), "max_iter", 5000)
    solver_options.SetOption(solver.solver_id(), "polish", True)
    solver_options.SetOption(solver.solver_id(), "polish_refine_iter", 3)
    solver_options.SetOption(solver.solver_id(), "warm_start", True)
    solver_options.SetOption(solver.solver_id(), "verbose", False)
    prog = MathematicalProgram()
    num_vars = plant.num_velocities() + plant.num_actuators()
    optimization_var = prog.NewContinuousVariables(num_vars, "q")
    equality_fn, inequality_fn, objective_fn = optimization_utilities.initialize_optimization(plant=plant)
    constraint_handle, objective_handle, prog = optimization_utilities.initialize_program(
        constraint_constants=constraint_constants,
        objective_constants=objective_constants,
        equality_functions=equality_fn,
        inequality_functions=inequality_fn,
        objective_functions=objective_fn,
        program=prog,
        optimization_size=(plant.num_velocities(), plant.num_actuators()),
    )
    # Create Isolated Update Function:
    update_optimization = lambda constraint_constants, objective_constants, program, solver, solver_options: optimization_utilities.update_program(
        constraint_constants,
        objective_constants,
        program,
        solver,
        solver_options,
        equality_fn,
        inequality_fn,
        objective_fn,
        (constraint_handle, objective_handle),
        (plant.num_velocities(), plant.num_actuators()),
    )

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

        # Left Arm:
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

        # OSC:
        time = context.get_time()
        a_1 = 1/4
        a_2 = 1/4
        rate_1 = a_1 * time
        rate_2 = a_2 * time
        r_1 = 0.05
        r_2 = 0.2
        xc = 0.3
        yc = 0.3

        x = xc + r_1 * np.cos(rate_1)
        y = yc + r_2 * np.sin(rate_2)

        dx = -r_1 * a_1 * np.sin(rate_1)
        dy = r_2 * a_2 * np.cos(rate_2)

        ddx = -r_1 * a_1 * a_1 * np.cos(rate_1)
        ddy = -r_2 * a_2 * a_2 * np.sin(rate_2)

        zero_vector = np.zeros((3,))
        ddx_desired = np.array([0, 0, 0, 0, ddx, ddy])
        dx_desired = np.array([0, 0, 0, 0, dx, dy])
        x_desired = np.array([0, 0, 0, 0.5, x, y])
        kp = 100
        kd = 2 * np.sqrt(kp)
        task_position = task_space_transform.translation()
        task_velocity = (spatial_velocity_jacobian @ qd)[3:]
        x_task = np.concatenate([zero_vector, task_position])
        dx_task = np.concatenate([zero_vector, task_velocity])
        control_desired = ddx_desired + kp * (x_desired - x_task) + kd * (dx_desired - dx_task)

        # Pack Optimization Constants:
        constraint_constants = (M, C, tau_g, B_left)
        objective_constants = (
            spatial_velocity_jacobian,
            bias_spatial_acceleration,
            control_desired,
        )

        # Solve Optimization:
        solution = update_optimization(
            constraint_constants=constraint_constants,
            objective_constants=objective_constants,
            program=prog,
            solver=solver,
            solver_options=solver_options,
        )

        # Unpack Optimization Solution:
        u = solution.GetSolution(optimization_var)[plant.num_velocities():]

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
