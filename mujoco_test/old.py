import numpy as np

from pydrake.common.value import Value
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.framework import (
    DiagramBuilder,
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import digit_api

import pdb


class digit_module(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Constants:
        self.NUM_MOTORS = 20
        self.NUM_JOINTS = 10
        self.control_rate = 1.0 / 1000.0
        self.fallback_opmode = 0

        # Define abstract states:
        motor_output_size = np.zeros((self.NUM_MOTORS,))
        joint_output_size = np.zeros((self.NUM_JOINTS,))
        motor_output_init = Value[BasicVector_[float]](motor_output_size)
        joint_output_init = Value[BasicVector_[float]](joint_output_size)
        self.motor_position_index = self.DeclareAbstractState(motor_output_init)
        self.motor_velocity_index = self.DeclareAbstractState(motor_output_init)
        self.motor_torque_index = self.DeclareAbstractState(motor_output_init)
        self.joint_position_index = self.DeclareAbstractState(joint_output_init)
        self.joint_velocity_index = self.DeclareAbstractState(joint_output_init)

        # Define output ports:
        self.motor_position_output = self.DeclareVectorOutputPort(
            "motor_position",
            self.NUM_MOTORS,
            self.get_motor_positions,
            {self.abstract_state_ticket(self.motor_position_index)},
        ).get_index()
        self.motor_velocity_output = self.DeclareVectorOutputPort(
            "motor_velocity",
            self.NUM_MOTORS,
            self.get_motor_velocities,
            {self.abstract_state_ticket(self.motor_velocity_index)},
        ).get_index()
        self.motor_torque_output = self.DeclareVectorOutputPort(
            "motor_torque",
            self.NUM_MOTORS,
            self.get_motor_torques,
            {self.abstract_state_ticket(self.motor_torque_index)},
        ).get_index()
        self.joint_position_output = self.DeclareVectorOutputPort(
            "joint_position",
            self.NUM_JOINTS,
            self.get_joint_positions,
            {self.abstract_state_ticket(self.joint_position_index)},
        ).get_index()
        self.joint_velocity_output = self.DeclareVectorOutputPort(
            "joint_velocity",
            self.NUM_JOINTS,
            self.get_joint_velocities,
            {self.abstract_state_ticket(self.joint_velocity_index)},
        ).get_index()

        # Define input ports:
        self.torque_command_input = self.DeclareVectorInputPort("torque_command", self.NUM_MOTORS).get_index()
        self.velocity_command_input = self.DeclareVectorInputPort("velocity_command", self.NUM_MOTORS).get_index()
        self.damping_command_input = self.DeclareVectorInputPort("damping_command", self.NUM_MOTORS).get_index()

        def update_output_ports(context, event):
            # Get observations from Digit API:
            motor_position = digit_api.get_actuated_joint_position()
            motor_velocity = digit_api.get_actuated_joint_velocity()
            motor_torque = digit_api.get_actuated_joint_torque()
            joint_position = digit_api.get_unactuated_joint_position()
            joint_velocity = digit_api.get_unactuated_joint_velocity()

            # Update abstract states:
            abstract_state = context.get_mutable_abstract_state(self.motor_position_index)
            abstract_state.set_value(motor_position)
            abstract_state = context.get_mutable_abstract_state(self.motor_velocity_index)
            abstract_state.set_value(motor_velocity)
            abstract_state = context.get_mutable_abstract_state(self.motor_torque_index)
            abstract_state.set_value(motor_torque)
            abstract_state = context.get_mutable_abstract_state(self.joint_position_index)
            abstract_state.set_value(joint_position)
            abstract_state = context.get_mutable_abstract_state(self.joint_velocity_index)
            abstract_state.set_value(joint_velocity)

        self.DeclarePeriodicEvent(
            period_sec=self.control_rate,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=update_output_ports,
            ),
        )

        def send_command(context, event):
            # Get command from input ports:
            torque_command = self.get_input_port(self.torque_command_input).Eval(context)
            velocity_command = self.get_input_port(self.velocity_command_input).Eval(context)
            damping_command = self.get_input_port(self.damping_command_input).Eval(context)
            apply_command = True

            # Send command:
            command = np.array([torque_command, velocity_command, damping_command]).T
            digit_api.send_command(command, self.fallback_opmode, apply_command)

        self.DeclarePeriodicEvent(
            period_sec=self.control_rate,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=send_command,
            ),
        )

    # Create generic output callbacks:
    def get_motor_positions(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.motor_position_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_motor_velocities(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.motor_velocity_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_motor_torques(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.motor_torque_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_joint_positions(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.joint_position_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_joint_velocities(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.joint_velocity_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    # Driver functions:
    @staticmethod
    def wait_for_connection():
        digit_api.wait_for_connection()

    @staticmethod
    def initialize_communication(
        publisher_address="127.0.0.1",
        subscriber_port=25501,
        publisher_port=25500,
    ):
        digit_api.initialize_communication(
            publisher_address,
            subscriber_port,
            publisher_port,
        )


class pd_controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Constants:
        self.NUM_MOTORS = 20
        self.NUM_JOINTS = 10
        self.control_rate = 1.0 / 1000.0
        self.fallback_opmode = 0

        # Safety Limits:
        self.torque_limits = digit_api.get_torque_limits()
        self.damping_limits = digit_api.get_damping_limits()
        self.velocity_limits = digit_api.get_velocity_limits()

        # Define abstract states:
        command_output_size = np.zeros((self.NUM_MOTORS,))
        command_output_init = Value[BasicVector_[float]](command_output_size)
        self.torque_command_index = self.DeclareAbstractState(command_output_init)
        self.velocity_command_index = self.DeclareAbstractState(command_output_init)
        self.damping_command_index = self.DeclareAbstractState(command_output_init)

        # Define output ports:
        self.torque_command_output = self.DeclareVectorOutputPort(
            "torque_command",
            self.NUM_MOTORS,
            self.get_torque_command,
            {self.abstract_state_ticket(self.torque_command_index)},
        ).get_index()
        self.velocity_command_output = self.DeclareVectorOutputPort(
            "velocity_command",
            self.NUM_MOTORS,
            self.get_velocity_command,
            {self.abstract_state_ticket(self.velocity_command_index)},
        ).get_index()
        self.damping_command_output = self.DeclareVectorOutputPort(
            "damping_command",
            self.NUM_MOTORS,
            self.get_damping_command,
            {self.abstract_state_ticket(self.damping_command_index)},
        ).get_index()

        # Define input ports:
        self.motor_position_input = self.DeclareVectorInputPort("motor_position", self.NUM_MOTORS).get_index()
        self.motor_velocity_input = self.DeclareVectorInputPort("motor_velocity", self.NUM_MOTORS).get_index()
        self.motor_torque_input = self.DeclareVectorInputPort("motor_torque", self.NUM_MOTORS).get_index()
        self.target_position_input = self.DeclareVectorInputPort("target_position", self.NUM_MOTORS).get_index()
        self.joint_position_input = self.DeclareVectorInputPort("joint_position", self.NUM_JOINTS).get_index()
        self.joint_velocity_input = self.DeclareVectorInputPort("joint_velocity", self.NUM_JOINTS).get_index()

        def update_output_ports(context, event):
            self.calculate_command(context, event)

        self.DeclarePeriodicEvent(
            period_sec=self.control_rate,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=update_output_ports,
            ),
        )

    # Create generic output callbacks:
    def get_torque_command(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.torque_command_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_velocity_command(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.velocity_command_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    def get_damping_command(self, context, output):
        abstract_state = context.get_mutable_abstract_state(self.damping_command_index)
        abstract_value = abstract_state.get_mutable_value()
        output.SetFromVector(abstract_value.get_mutable_value())

    # PD Control:
    def calculate_command(self, context, event):
        # Get observations from input ports:
        motor_position = self.get_input_port(self.motor_position_input).Eval(context)
        motor_velocity = self.get_input_port(self.motor_velocity_input).Eval(context)
        motor_torque = self.get_input_port(self.motor_torque_input).Eval(context)
        joint_position = self.get_input_port(self.joint_position_input).Eval(context)
        joint_velocity = self.get_input_port(self.joint_velocity_input).Eval(context)

        # Get target position:
        target_position = self.get_input_port(self.target_position_input).Eval(context)

        # Calculate control:
        desired_torques = []
        desired_velocities = []
        desired_damping = []
        for i in range(self.NUM_MOTORS):
            desired_torques.append(150.0 * (target_position[i] - motor_position[i]))
            desired_velocities.append(0.0)
            desired_damping.append(0.75 * self.damping_limits[i])

        # Create Command Array:
        desired_torques = np.asarray(desired_torques)
        desired_velocities = np.asarray(desired_velocities)
        desired_damping = np.asarray(desired_damping)

        # Update abstract states:
        abstract_state = context.get_mutable_abstract_state(self.torque_command_index)
        abstract_state.set_value(desired_torques)
        abstract_state = context.get_mutable_abstract_state(self.velocity_command_index)
        abstract_state.set_value(desired_velocities)
        abstract_state = context.get_mutable_abstract_state(self.damping_command_index)
        abstract_state.set_value(desired_damping)


def main(argv=None):
    # Create block diagram:
    builder = DiagramBuilder()

    # Digit Module:
    digit_driver = digit_module()
    digit_driver.initialize_communication()

    # Wait for lowlevelapi connection:
    digit_driver.wait_for_connection()

    digit = builder.AddSystem(digit_driver)

    # PD Controller:
    controller_driver = pd_controller()
    controller = builder.AddSystem(controller_driver)

    # Target Position:
    target_position = np.array([
        -0.0462933, -0.0265814, 0.19299, -0.3, -0.0235182,
        -0.0571617, 0.0462933, 0.0265814, -0.19299, 0.3,
        -0.0235182, 0.0571617, -0.3, 0.943845, 0.0,
        0.3633, 0.3, -0.943845, 0.0, -0.3633,
    ])
    target_position_source = builder.AddSystem(
        ConstantVectorSource(target_position),
    )

    # Connect Diagram:
    # Controller Inputs:
    builder.Connect(
        target_position_source.get_output_port(),
        controller.get_input_port(controller_driver.target_position_input),
    )
    builder.Connect(
        digit.get_output_port(digit_driver.motor_position_output),
        controller.get_input_port(controller_driver.motor_position_input),
    )
    builder.Connect(
        digit.get_output_port(digit_driver.motor_velocity_output),
        controller.get_input_port(controller_driver.motor_velocity_input),
    )
    builder.Connect(
        digit.get_output_port(digit_driver.motor_torque_output),
        controller.get_input_port(controller_driver.motor_torque_input),
    )
    builder.Connect(
        digit.get_output_port(digit_driver.joint_position_output),
        controller.get_input_port(controller_driver.joint_position_input),
    )
    builder.Connect(
        digit.get_output_port(digit_driver.joint_velocity_output),
        controller.get_input_port(controller_driver.joint_velocity_input),
    )

    # Digit Inputs:
    builder.Connect(
        controller.get_output_port(controller_driver.torque_command_output),
        digit.get_input_port(digit_driver.torque_command_input),
    )
    builder.Connect(
        controller.get_output_port(controller_driver.velocity_command_output),
        digit.get_input_port(digit_driver.velocity_command_input),
    )
    builder.Connect(
        controller.get_output_port(controller_driver.damping_command_output),
        digit.get_input_port(digit_driver.damping_command_input),
    )

    # Build Diagram:
    diagram = builder.Build()

    # Create Simulator:
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    # Run Control:
    target_time = 10.0
    simulator.AdvanceTo(target_time)


if __name__ == "__main__":
    main()