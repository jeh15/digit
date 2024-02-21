import numpy as np

from pydrake.common.value import Value
from pydrake.systems.framework import (
    BasicVector,
    LeafSystem,
    PublishEvent,
    TriggerType,
)

from pydrake.multibody.plant import MultibodyPlant

from controller_module import make_solution_wrapper_value, SolutionWrapper


class PlantPortProcessor(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        update_rate: float = 1.0 / 1000.0,
    ):
        super().__init__()
        self.plant = plant
        self.update_rate = update_rate
        self.num_actuators = plant.num_actuators()

        # Abstract State:
        self.torque_size = np.zeros((self.num_actuators,))
        self.torque_index = self.DeclareAbstractState(
            Value[BasicVector](self.torque_size),
        )

        # Input Port: Optimization Solution
        self.solution_port = self.DeclareAbstractInputPort(
            "solution",
            make_solution_wrapper_value(SolutionWrapper()),
        ).get_index()

        # Output Port: Plant Torque
        def output_callback(context, output):
            torque_state = context.get_abstract_state(
                self.torque_index,
            )
            torque = torque_state.get_value().get_value()
            output.SetFromVector(torque)

        self.plant_torque_port = self.DeclareVectorOutputPort(
            "torque",
            BasicVector(self.num_actuators),
            output_callback,
        ).get_index()

        def on_initialization(context, event):
            # Process Optimization Solution:
            torque_state = context.get_mutable_abstract_state(
                self.torque_index,
            )
            torque_state.set_value(
                np.zeros((self.num_actuators,)),
            )

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        # Declare Periodic Event: Process Optimization Solution
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
        # Get Optimization Solution:
        solution = self.get_input_port(
            self.solution_port,
        ).Eval(context)

        # Process Optimization Solution:
        torque_state = context.get_mutable_abstract_state(
            self.torque_index,
        )
        torque_state.set_value(
            solution.torque,
        )
