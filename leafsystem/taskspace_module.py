import numpy as np

from pydrake.common.value import Value
from pydrake.systems.framework import (
    BasicVector,
    LeafSystem,
    PublishEvent,
    TriggerType,
    Context,
)
from pydrake.multibody.plant import MultibodyPlant

import dynamics_utilities


class TaskSpace(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
    ):
        LeafSystem.__init__(self)
        # Store Plant and Create Context:
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()

        # Parameters:
        self.update_rate = 1.0 / 1000.0

        # Abstract States: Task Space Matrices
        self.task_transform_rotation_size = np.zeros(12)
        self.task_transform_translation_size = np.zeros(9)
        self.task_jacobian_size = np.zeros(612)
        self.task_bias_size = np.zeros(18)
        self.task_transform_rotation_index = self.DeclareAbstractState(
            Value[BasicVector](self.task_transform_rotation_size)
        )
        self.task_transform_translation_index = self.DeclareAbstractState(
            Value[BasicVector](self.task_transform_translation_size)
        )
        self.task_jacobian_index = self.DeclareAbstractState(
            Value[BasicVector](self.task_jacobian_size)
        )
        self.task_bias_index = self.DeclareAbstractState(
            Value[BasicVector](self.task_bias_size)
        )

        # Output Port: Task Space Matrices
        self.task_transform_rotation_port = self.DeclareVectorOutputPort(
            "task_rotation",
            BasicVector(self.task_transform_rotation_size),
            self.rotation_output_callback,
            {self.abstract_state_ticket(self.task_transform_rotation_index)},
        ).get_index()
        self.task_transform_translation_port = self.DeclareVectorOutputPort(
            "task_translation",
            BasicVector(self.task_transform_translation_size),
            self.translation_output_callback,
            {self.abstract_state_ticket(self.task_transform_translation_index)},
        ).get_index()
        self.task_jacobian_port = self.DeclareVectorOutputPort(
            "task_jacobian",
            BasicVector(self.task_jacobian_size),
            self.jacobian_output_callback,
            {self.abstract_state_ticket(self.task_jacobian_index)},
        ).get_index()
        self.task_bias_port = self.DeclareVectorOutputPort(
            "task_bias",
            BasicVector(self.task_bias_size),
            self.bias_output_callback,
            {self.abstract_state_ticket(self.task_bias_index)},
        ).get_index()

        # Initialize:
        def on_initialization(context, event):
            self.update(context, event)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        # Declare Periodic Event: Update Task Space Matrices
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

    def rotation_output_callback(self, context, output):
        rotation_state = context.get_mutable_abstract_state(
            self.task_transform_rotation_index,
        )
        rotation = rotation_state.get_mutable_value().get_value()
        output.SetFromVector(
            rotation,
        )

    def translation_output_callback(self, context, output):
        translation_state = context.get_mutable_abstract_state(
            self.task_transform_translation_index,
        )
        translation = translation_state.get_mutable_value().get_value()
        output.SetFromVector(
            translation,
        )

    def jacobian_output_callback(self, context, output):
        jacobian_state = context.get_mutable_abstract_state(
            self.task_jacobian_index,
        )
        jacobian = jacobian_state.get_mutable_value().get_value()
        output.SetFromVector(
            jacobian,
        )

    def bias_output_callback(self, context, output):
        bias_state = context.get_mutable_abstract_state(
            self.task_bias_index,
        )
        bias = bias_state.get_mutable_value().get_value()
        output.SetFromVector(
            bias,
        )

    def update(self, context, event):
        # Update Task Space Matrices:
        task_transform, task_jacobian, task_bias = dynamics_utilities.calculate_taskspace(
            plant=self.plant,
            context=self.plant_context,
            body_name=[
                "base_link",
                "left-foot_link",
                "right-foot_link",
            ],
            base_body_name="world",
        )

        # Flatten:
        task_transform_rotation = np.asarray(
            list(
                map(lambda x: x.rotation().ToQuaternion().wxyz(), task_transform),
            )
        ).flatten()
        task_transform_translation = np.asarray(
            list(
                map(lambda x: x.translation(), task_transform),
            )
        ).flatten()
        task_jacobian = task_jacobian.flatten()

        # Set Abstract States:
        rotation_state = context.get_mutable_abstract_state(
            self.task_transform_rotation_index,
        )
        rotation_state.set_value(task_transform_rotation)
        translation_state = context.get_mutable_abstract_state(
            self.task_transform_translation_index,
        )
        translation_state.set_value(task_transform_translation)
        jacobian_state = context.get_mutable_abstract_state(
            self.task_jacobian_index,
        )
        jacobian_state.set_value(task_jacobian)
        bias_state = context.get_mutable_abstract_state(
            self.task_bias_index,
        )
        bias_state.set_value(task_bias)
