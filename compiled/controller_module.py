import numpy as np
import osqp

from pydrake.common.value import Value
from pydrake.common.eigen_geometry import Quaternion
from pydrake.systems.framework import (
    BasicVector,
    LeafSystem,
    PublishEvent,
    TriggerType,
)
from pydrake.multibody.plant import MultibodyPlant

import dynamics_utilities
import optimization_utilities
import model_utilities

from context_utilities import make_context_wrapper_value
from digit_utilities import DigitUtilities
from trajectory_module import make_trajectory_wrapper_value

import time


class OSC(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        digit_idx: DigitUtilities,
        constraint_frames: list
    ):
        LeafSystem.__init__(self)

        # Store Plant and Create Context:
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()

        # Digit Convinience Functions:
        self.digit_idx = digit_idx

        # Constraint Frames:
        self.constraint_frames = constraint_frames

        # Parameters:
        self.update_rate = 1.0 / 1000.0
        self.B = self.digit_idx.control_matrix

        # Abstract States: OSC Solution -- Output
        self.acceleration_size = np.zeros(plant.num_velocities())
        self.torque_size = np.zeros(plant.num_actuators())
        self.constraint_force_size = np.zeros(6)
        self.reaction_force_size = np.zeros(12)
        self.acceleration_index = self.DeclareAbstractState(
            Value[BasicVector](self.acceleration_size)
        )
        self.torque_index = self.DeclareAbstractState(
            Value[BasicVector](self.torque_size)
        )
        self.constraint_force_index = self.DeclareAbstractState(
            Value[BasicVector](self.constraint_force_size)
        )
        self.reaction_force_index = self.DeclareAbstractState(
            Value[BasicVector](self.reaction_force_size)
        )

        # Input Port: Task Space Matrices and Desired Acceleration
        self.task_jacobian_port = self.DeclareVectorInputPort(
            "task_jacobian", 1428,
        ).get_index()
        self.task_bias_port = self.DeclareVectorInputPort(
            "task_bias", 42,
        ).get_index()
        self.desired_ddx_port = self.DeclareVectorInputPort(
            "desired_ddx", 42,
        ).get_index()

        # Input Port: Plant Context
        self.plant_context_port = self.DeclareAbstractInputPort(
            "plant_context",
            make_context_wrapper_value(self.plant),
        ).get_index()

        # Output Port: OSC Torque Solution
        self.torque_port = self.DeclareVectorOutputPort(
            "torque",
            BasicVector(self.torque_size),
            self.output_callback,
        ).get_index()

        # Convenience Variables:
        self.dv_indx = self.acceleration_size.shape[0]
        self.u_indx = self.torque_size.shape[0] + self.dv_indx
        self.f_indx = self.constraint_force_size.shape[0] + self.u_indx

        # Declare Initialization Event: Initialize Optimization
        def on_initialization(context, event):
            self.initialize(context, event)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        # Declare Periodic Event: Solve Optimization
        def on_periodic(context, event):
            self.update(context, event)

        self.DeclarePeriodicEvent(
            period_sec=self.update_rate,
            offset_sec=self.update_rate,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
            ),
        )

    def output_callback(self, context, output):
        # Get Abstract States:
        torque_state = context.get_mutable_abstract_state(
            self.torque_index,
        )
        torque = torque_state.get_mutable_value().get_value()
        output.SetFromVector(torque)

    def initialize(self, context, event):
        # Initialize Static Parameters:
        optimization_size = (
            self.acceleration_size.shape[0],
            self.torque_size.shape[0],
            self.constraint_force_size.shape[0],
            self.reaction_force_size.shape[0],
        )

        # Calculate Dynamics:
        dynamics_terms = self.calculate_dynamics(
            plant_context=self.plant_context,
        )
        M, C, tau_g, H, H_bias = dynamics_terms

        # TODO(jeh15): Take this out of the initialize function:
        task_transform, task_jacobian, task_bias = dynamics_utilities.calculate_taskspace(
            plant=self.plant,
            context=self.plant_context,
            body_name=[
                "base_link",
                "left-foot_link",
                "right-foot_link",
                "left-hand_link",
                "right-hand_link",
                "left-elbow_link",
                "right-elbow_link",
            ],
            base_body_name="world",
        )

        ddx_desired = np.zeros(
            (task_jacobian.shape[0],)
        )

        weight = self.plant.CalcTotalMass(context=self.plant_context) * 9.81
        reaction_force = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, weight/2,
             0.0, 0.0, 0.0, 0.0, 0.0, weight/2],
        )
        previous_ground_reaction_forces = np.reshape(
            reaction_force,
            (2, 6),
        )[:, -1]

        # Initialize Abstract State:
        reaction_force_state = context.get_mutable_abstract_state(
            self.reaction_force_index,
        )
        reaction_force_state.set_value(reaction_force)

        constraint_constants = (
            M,
            C,
            tau_g,
            self.B,
            H,
            H_bias,
            previous_ground_reaction_forces,
        )

        objective_constants = (
            task_jacobian,
            task_bias,
            ddx_desired,
        )

        # Initialize Solver and Optimization:
        self.program = osqp.OSQP()
        equality_fn, inequality_fn, objective_fn = optimization_utilities.link_shared_library()

        self.program = optimization_utilities.initialize_program(
            constraint_constants=constraint_constants,
            objective_constants=objective_constants,
            program=self.program,
            equality_functions=equality_fn,
            inequality_functions=inequality_fn,
            objective_functions=objective_fn,
            optimization_size=optimization_size,
        )

        # Create Isolated Update Function:
        self.update_optimization = lambda constraint_constants, objective_constants, program: optimization_utilities.update_program(
            constraint_constants,
            objective_constants,
            program,
            equality_fn,
            inequality_fn,
            objective_fn,
            optimization_size,
        )

    def update(self, context, event):
        # Update Plant Context:
        self.plant_context = self.get_input_port(
            self.plant_context_port,
        ).Eval(context).context

        # Get Input Ports: Task Space Matrices and Desired Acceleration
        task_jacobian = self.get_input_port(
            self.task_jacobian_port,
        ).Eval(context)
        task_bias = self.get_input_port(
            self.task_bias_port,
        ).Eval(context)
        ddx_desired = self.get_input_port(
            self.desired_ddx_port,
        ).Eval(context)

        # Reshape Matrices:
        task_jacobian = np.reshape(
            np.asarray(task_jacobian),
            (-1, 34),
        )
        task_bias = np.asarray(task_bias)
        ddx_desired = np.asarray(ddx_desired)

        # Calculate Dynamics:
        dynamics_terms = self.calculate_dynamics(
            plant_context=self.plant_context,
        )
        M, C, tau_g, H, H_bias = dynamics_terms

        # Get Previous Ground Reaction Forces:
        abstract_state = context.get_mutable_abstract_state(
            self.reaction_force_index,
        )
        previous_ground_reaction_forces = np.reshape(
            abstract_state.get_mutable_value().get_value(),
            (2, 6),
        )[:, -1]

        constraint_constants = (
            M,
            C,
            tau_g,
            self.B,
            H,
            H_bias,
            previous_ground_reaction_forces,
        )

        objective_constants = (
            task_jacobian,
            task_bias,
            ddx_desired,
        )

        start_time = time.time()
        solution, self.program = self.update_optimization(
            constraint_constants=constraint_constants,
            objective_constants=objective_constants,
            program=self.program,
        )
        end_time = time.time()

        # Unpack Optimization Solution:
        accelerations = solution.x[:self.dv_indx]
        torque = solution.x[self.dv_indx:self.u_indx]
        constraint_force = solution.x[self.u_indx:self.f_indx]
        reaction_force = solution.x[self.f_indx:]

        # Update Abstract States:
        accelerations_state = context.get_mutable_abstract_state(
            self.acceleration_index,
        )
        accelerations_state.set_value(accelerations)
        torque_state = context.get_mutable_abstract_state(
            self.torque_index,
        )
        torque_state.set_value(torque)
        constraint_force_state = context.get_mutable_abstract_state(
            self.constraint_force_index,
        )
        constraint_force_state.set_value(constraint_force)
        reaction_force_state = context.get_mutable_abstract_state(
            self.reaction_force_index,
        )
        reaction_force_state.set_value(reaction_force)

    def calculate_dynamics(
        self,
        plant_context,
    ):
        M, C, tau_g = dynamics_utilities.get_dynamics(
            plant=self.plant,
            context=plant_context,
        )
        H, H_bias = dynamics_utilities.calculate_kinematic_constraints(
            plant=self.plant,
            context=plant_context,
            constraint_frames=self.constraint_frames,
        )
        H = np.concatenate(
            [model_utilities.achilles_rod_constraint(), H],
            axis=0,
        )
        H_bias = np.concatenate(
            [
                np.zeros(
                    (model_utilities.achilles_rod_constraint().shape[0],)
                ),
                H_bias,
            ],
            axis=0,
        )

        dynamics_terms = (M, C, tau_g, H, H_bias)

        return dynamics_terms


class PID(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        digit_idx: DigitUtilities,
    ):
        LeafSystem.__init__(self)
        # Store Plant and Create Context:
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()

        # Digit Convinience Functions:
        self.digit_idx = digit_idx

        # Parameters:
        self.update_rate = 1.0 / 1000.0

        # Poor Mans Timer:
        self.index = 0.0

        # Abstract States: Control Input
        self.control_input_size = np.zeros(42)
        self.control_input_index = self.DeclareAbstractState(
            Value[BasicVector](self.control_input_size)
        )

        # Input Port: Task Space Matrices
        self.task_jacobian_port = self.DeclareVectorInputPort(
            "task_jacobian", 1428,
        ).get_index()
        self.task_transform_rotation_port = self.DeclareVectorInputPort(
            "task_transform_rotation", 28,
        ).get_index()
        self.task_transform_translation_port = self.DeclareVectorInputPort(
            "task_transform_translation", 21,
        ).get_index()

        # Input Port: Plant Context
        self.plant_context_port = self.DeclareAbstractInputPort(
            "plant_context",
            make_context_wrapper_value(self.plant),
        ).get_index()

        # Input Port: Trajectory
        self.trajectory_port = self.DeclareAbstractInputPort(
            "trajectory",
            make_trajectory_wrapper_value(),
        ).get_index()

        # Output Port: Control Input
        self.control_port = self.DeclareVectorOutputPort(
            "control_input",
            BasicVector(self.control_input_size),
            self.output_callback,
            {self.all_sources_ticket()}
        ).get_index()

        # Declare Initialization Event:
        def on_initialization(context, event):
            control_input = np.zeros_like(self.control_input_size)
            control_input_state = context.get_mutable_abstract_state(
                self.control_input_index,
            )
            control_input_state.set_value(control_input)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        # Declare Periodic Event: Solve for Control Input
        def on_periodic(context, event):
            self.update(context, event)

        self.DeclarePeriodicEvent(
            period_sec=self.update_rate,
            offset_sec=self.update_rate,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
            ),
        )

    def output_callback(self, context, output):
        # Get Abstract States:
        control_input_state = context.get_mutable_abstract_state(
            self.control_input_index,
        )
        control_input = control_input_state.get_mutable_value().get_value()
        output.SetFromVector(control_input)

    def update(self, context, event):
        # Update Plant Context:
        self.plant_context = self.get_input_port(
            self.plant_context_port,
        ).Eval(context).context

        # Get Input Ports: Trajectory
        trajectory = self.get_input_port(
            self.trajectory_port,
        ).Eval(context)

        # Get Input Ports: Task Space Matrices
        task_jacobian = self.get_input_port(
            self.task_jacobian_port,
        ).Eval(context)
        task_transform_rotation = self.get_input_port(
            self.task_transform_rotation_port,
        ).Eval(context)
        task_transform_translation = self.get_input_port(
            self.task_transform_translation_port,
        ).Eval(context)

        # Reshape Matrices:
        task_jacobian = np.reshape(
            np.asarray(task_jacobian),
            (-1, 34),
        )
        task_transform_rotation = np.reshape(
            np.asarray(task_transform_rotation),
            (-1, 4),
        )
        task_transform_translation = np.reshape(
            np.asarray(task_transform_translation),
            (-1, 3),
        )

        # Get Plant States:
        qd = self.plant.GetVelocities(self.plant_context)

        # Calculate Desired Control:
        # Static Base Tracking:
        # kp_position_base = 200.0
        # kd_position_base = 2 * np.sqrt(kp_position_base)
        # kp_rotation_base = 200.0
        # kd_rotation_base = 2 * np.sqrt(kp_rotation_base)
        # kp_position_feet = 0.0
        # kd_position_feet = 2 * np.sqrt(kp_position_feet)
        # kp_rotation_feet = 100.0
        # kd_rotation_feet = 2 * np.sqrt(kp_rotation_feet)

        # Base Tracking:
        kp_position_base = 200.0
        kd_position_base = 2 * np.sqrt(kp_position_base)
        kp_rotation_base = 500.0
        kd_rotation_base = 200 * np.sqrt(kp_rotation_base)
        # kp_rotation_base = 150.0
        # kd_rotation_base = 2 * np.sqrt(kp_rotation_base)

        # Feet Tracking:
        kp_position_feet = 0.0
        kd_position_feet = 2 * np.sqrt(kp_position_feet)
        kp_rotation_feet = 100.0
        kd_rotation_feet = 2 * np.sqrt(kp_rotation_feet)

        # Hand Tracking:
        kp_position_hands = 500.0
        kd_position_hands = 2 * np.sqrt(kp_position_hands)
        kp_rotation_hands = 0.0
        kd_rotation_hands = 2 * np.sqrt(kp_rotation_hands)

        # Elbow Tracking:
        kp_position_elbows = 100.0
        kd_position_elbows = 2 * np.sqrt(kp_position_elbows)
        kp_rotation_elbows = 0.0
        kd_rotation_elbows = 2 * np.sqrt(kp_rotation_elbows)

        control_gains = [
            [kp_position_base, kd_position_base,
             kp_rotation_base, kd_rotation_base],
            [kp_position_feet, kd_position_feet,
             kp_rotation_feet, kd_rotation_feet],
            [kp_position_feet, kd_position_feet,
             kp_rotation_feet, kd_rotation_feet],
            [kp_position_hands, kd_position_hands,
             kp_rotation_hands, kd_rotation_hands],
            [kp_position_hands, kd_position_hands,
             kp_rotation_hands, kd_rotation_hands],
            [kp_position_elbows, kd_position_elbows,
             kp_rotation_elbows, kd_rotation_elbows],
            [kp_position_elbows, kd_position_elbows,
             kp_rotation_elbows, kd_rotation_elbows],
        ]

        position_target = [
            [trajectory.base_trajectory['ddx'], trajectory.base_trajectory['dx'], trajectory.base_trajectory['x']],
            [trajectory.left_foot_trajectory['ddx'], trajectory.left_foot_trajectory['dx'], trajectory.left_foot_trajectory['x']],
            [trajectory.right_foot_trajectory['ddx'], trajectory.right_foot_trajectory['dx'], trajectory.right_foot_trajectory['x']],
            [trajectory.left_hand_trajectory['ddx'], trajectory.left_hand_trajectory['dx'], trajectory.left_hand_trajectory['x']],
            [trajectory.right_hand_trajectory['ddx'], trajectory.right_hand_trajectory['dx'], trajectory.right_hand_trajectory['x']],
            [trajectory.left_elbow_trajectory['ddx'], trajectory.left_elbow_trajectory['dx'], trajectory.left_elbow_trajectory['x']],
            [trajectory.right_elbow_trajectory['ddx'], trajectory.right_elbow_trajectory['dx'], trajectory.right_elbow_trajectory['x']],
        ]

        rotation_target = [
            [trajectory.base_trajectory['ddw'], trajectory.base_trajectory['dw'], trajectory.base_trajectory['w']],
            [trajectory.left_foot_trajectory['ddw'], trajectory.left_foot_trajectory['dw'], trajectory.left_foot_trajectory['w']],
            [trajectory.right_foot_trajectory['ddw'], trajectory.right_foot_trajectory['dw'], trajectory.right_foot_trajectory['w']],
            [trajectory.left_hand_trajectory['ddw'], trajectory.left_hand_trajectory['dw'], trajectory.left_hand_trajectory['w']],
            [trajectory.right_hand_trajectory['ddw'], trajectory.right_hand_trajectory['dw'], trajectory.right_hand_trajectory['w']],
            [trajectory.left_elbow_trajectory['ddw'], trajectory.left_elbow_trajectory['dw'], trajectory.left_elbow_trajectory['w']],
            [trajectory.right_elbow_trajectory['ddw'], trajectory.right_elbow_trajectory['dw'], trajectory.right_elbow_trajectory['w']],
        ]

        task_J = np.split(task_jacobian, 7)
        task_position = np.split(task_transform_translation, 7)
        task_rotation = np.split(task_transform_rotation, 7)

        loop_iterables = zip(
            task_position,
            task_rotation,
            task_J,
            position_target,
            rotation_target,
            control_gains,
        )

        # Calculate Desired Control:
        control_input = []
        for position, rotation, J, x_target, w_target, gains in loop_iterables:
            position = position.flatten()
            rotation = rotation.flatten()
            task_velocity = J @ qd
            task_rotation = Quaternion(wxyz=rotation)
            target_rotation = Quaternion(wxyz=w_target[2])
            # From Ickes, B. P. (1970): For control purposes the last three elements of the quaternion define the roll, pitch, and yaw rotational errors.
            rotation_error = target_rotation.multiply(task_rotation.conjugate()).xyz()
            position_control = x_target[0] + gains[1] * (x_target[1] - task_velocity[3:]) + gains[0] * (x_target[2] - position)
            rotation_control = w_target[0] + gains[2] * (w_target[1] - task_velocity[:3]) + gains[3] * (rotation_error)
            control_input.append(
                np.concatenate([rotation_control, position_control])
            )

        # Desired Accelerations:
        control_input = np.concatenate(control_input, axis=0)

        # Update Abstract States:
        control_input_state = context.get_mutable_abstract_state(
            self.control_input_index,
        )
        control_input_state.set_value(control_input)
