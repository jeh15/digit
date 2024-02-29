import numpy as np

from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
)
from pydrake.multibody.plant import MultibodyPlant

import digit_api
from digit_utilities import DigitUtilities
from context_utilities import make_context_wrapper_value
from safety_module import make_command_wrapper_value, CommandWrapper


class AgilityContextSystem(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        digit_idx: DigitUtilities,
    ):
        super().__init__()
        self.plant = plant
        self.digit_idx = digit_idx

        def calc_context(context, output):
            # Get observations from Digit API:
            base_rotation = digit_api.get_base_orientation()
            base_position = digit_api.get_base_position()
            base_angular_velocity = digit_api.get_base_angular_velocity()
            base_linear_velocity = digit_api.get_base_linear_velocity()
            motor_position = digit_api.get_actuated_joint_position()
            motor_velocity = digit_api.get_actuated_joint_velocity()
            joint_position = digit_api.get_unactuated_joint_position()
            joint_velocity = digit_api.get_unactuated_joint_velocity()
            # Transform observations to Drake's convention:
            base_q = np.concatenate([base_rotation, base_position])
            base_qd = np.concatenate([
                base_angular_velocity, base_linear_velocity,
            ])
            q = self.digit_idx.joint_map(
                motor_position, joint_position, base_q,
            )
            qd = self.digit_idx.joint_map(
                motor_velocity, joint_velocity, base_qd,
            )
            x = np.concatenate([q, qd])
            # Get Drake's context:
            wrapper = output.get_mutable_value()
            plant_context = wrapper.context
            # Set Drake's context:
            plant.SetPositionsAndVelocities(plant_context, x)
            if context.get_time() > 3.01:
                pass

        self.plant_context_port = self.DeclareAbstractOutputPort(
            "plant_context",
            alloc=lambda: make_context_wrapper_value(plant),
            calc=calc_context,
        ).get_index()


class AgilityPublisher(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        digit_idx: DigitUtilities,
        update_rate: float = 1.0 / 1000.0,
    ):
        super().__init__()
        self.plant = plant
        self.digit_idx = digit_idx
        self.num_actuators = self.plant.num_actuators()

        # Update Rate:
        self.update_rate = update_rate

        # Input Port: Command Struct
        self.command_port = self.DeclareAbstractInputPort(
            "command",
            make_command_wrapper_value(CommandWrapper()),
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

    def update(self, context, event):
        # Get input:
        command = self.get_input_port(
            self.command_port,
        ).Eval(context)

        torque_command = self.digit_idx.actuation_map(command.torque_command)
        velocity_command = command.velocity_command
        damping_command = command.damping_command
        fallback_opmode = command.fallback_opmode
        command = np.array([torque_command, velocity_command, damping_command]).T
        # Disabled = 0,
        # Damping = 1,
        # Locomotion = 2,
        digit_api.send_command(command, fallback_opmode, True)
