from dataclasses import dataclass, field

import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
)
from pydrake.multibody.plant import MultibodyPlant

from digit_utilities import DigitUtilities
from context_utilities import make_context_wrapper_value
from controller_module import make_solution_wrapper_value, SolutionWrapper
from websocket_module import make_message_wrapper_value, MessageWrapper


@dataclass
class CommandWrapper:
    torque_command: np.ndarray = field(default_factory=lambda: np.zeros(20))
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(20))
    damping_command: np.ndarray = field(default_factory=lambda: np.zeros(20))
    fallback_opmode: int = 1


def make_command_wrapper_value(command: CommandWrapper):
    return AbstractValue.Make(command)


class SafetyController(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        digit_idx: DigitUtilities,
        update_rate: float = 1.0 / 1000.0,
    ):
        LeafSystem.__init__(self)
        self.plant = plant
        self.update_rate = update_rate

        # TODO(jeh15) This needs be taken out. Make a dedicated low-level-api trigger:
        self.warmup_time = 1.0

        # Store Plant and Create Context:
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        self.num_actuators = self.plant.num_actuators()

        # Digit Convinience Functions:
        self.digit_idx = digit_idx

        # Abstract States:
        self.command_state = self.DeclareAbstractState(
            make_command_wrapper_value(CommandWrapper()),
        )
        self.message_state = self.DeclareAbstractState(
            make_message_wrapper_value(''),
        )

        # Input Port: Plant Context
        self.plant_context_port = self.DeclareAbstractInputPort(
            "plant_context",
            make_context_wrapper_value(self.plant),
        ).get_index()

        # Input Port: Optimization Solution
        self.solution_port = self.DeclareAbstractInputPort(
            "solution",
            make_solution_wrapper_value(SolutionWrapper()),
        ).get_index()

        # Output Port: Controller Command
        self.command_port = self.DeclareAbstractOutputPort(
            name="command",
            alloc=lambda: make_command_wrapper_value(CommandWrapper()),
            calc=self.command_output_callback,
        ).get_index()

        # Output Port: Safety Status
        self.message_port = self.DeclareAbstractOutputPort(
            "message",
            alloc=lambda: make_message_wrapper_value(''),
            calc=self.message_output_callback,
        ).get_index()

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

    def command_output_callback(self, context, output):
        command_state = context.get_mutable_abstract_state(
            self.command_state,
        )
        command = command_state.get_value()
        output.set_value(
            command
        )

    def message_output_callback(self, context, output):
        message_state = context.get_mutable_abstract_state(
            self.message_state,
        )
        message = message_state.get_value()
        output.set_value(
            message
        )

    def update(self, context, event):
        # Get inputs:
        plant_context = self.get_input_port(
            self.plant_context_port,
        ).Eval(context).context

        solution = self.get_input_port(
            self.solution_port,
        ).Eval(context)

        # Get Desired Torques:
        leg_torques = np.concatenate(
            [
                solution.torque[self.digit_idx.actuation_idx['left_leg']],
                solution.torque[self.digit_idx.actuation_idx['right_leg']],
            ]
        )
        arm_torques = np.concatenate(
            [
                solution.torque[self.digit_idx.actuation_idx['left_arm']],
                solution.torque[self.digit_idx.actuation_idx['right_arm']],
            ]
        )

        # Get joints:
        q = self.plant.GetPositions(plant_context)
        qd = self.plant.GetVelocities(plant_context)

        # Joint limits:
        q_min = self.plant.GetPositionLowerLimits()
        q_max = self.plant.GetPositionUpperLimits()
        qd_lim = (60.0 * np.pi / 180.0) * np.ones(self.plant.num_velocities())

        # Torque Limits:
        leg_torque_limit = np.array(
            [116, 70, 206, 220, 35, 35],
            dtype=np.float64,
        )
        arm_torque_limit = np.array(
            [35, 35, 35, 35],
            dtype=np.float64,
        )

        # Check joint limits:
        message = ''

        # TODO(jeh15) This needs be taken out. Make a dedicated low-level-api trigger:
        if context.get_time() > self.warmup_time and context.get_time() < self.warmup_time + 0.1:
            message = 'low-level-api'

        # if np.any(q < q_min) or np.any(q > q_max):
        #     print('Joint Limit Exceeded')
        #     message = 'shutdown'

        # if np.any(np.abs(qd) > qd_lim):
        #     print('Velocity Limit Exceeded')
        #     message = 'shutdown'

        if np.any(np.abs(leg_torques) > np.tile(leg_torque_limit, 2)):
            print('Leg Torque Limit Exceeded')
            message = 'shutdown'

        if np.any(np.abs(arm_torques) > np.tile(arm_torque_limit, 2)):
            print('Arm Torque Limit Exceeded')
            message = 'shutdown'

        if message == 'shutdown':
            print('Safety Shutdown')
            command = CommandWrapper()
        else:
            command = CommandWrapper(
                torque_command=solution.torque,
                velocity_command=np.zeros((self.num_actuators,)),
                damping_command=0.75 * np.ones((self.num_actuators,)),
                fallback_opmode=0,
            )

        # Set Abstract States:
        command_state = context.get_mutable_abstract_state(
            self.command_state,
        )
        command_state.set_value(
            command
        )
        message_state = context.get_mutable_abstract_state(
            self.message_state,
        )
        message_state.set_value(
            MessageWrapper(message)
        )


