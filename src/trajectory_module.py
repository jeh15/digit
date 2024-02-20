from typing import Any
from dataclasses import dataclass, field, fields

import numpy as np
import numpy.typing as npt

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import (
    LeafSystem,
    Context,
    PublishEvent,
    TriggerType,
)
from pydrake.multibody.plant import MultibodyPlant

# Trajectory Type:
Trajectory = dict[str, npt.NDArray[np.float64]]

default_initial_trajectory = {
    'x': np.zeros(3),
    'dx': np.zeros(3),
    'ddx': np.zeros(3),
    'w': np.zeros(4),
    'dw': np.zeros(3),
    'ddw': np.zeros(3),
}


@dataclass
class TrajectoryWrapper:
    base_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    left_foot_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    right_foot_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    left_hand_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    right_hand_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    left_elbow_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )
    right_elbow_trajectory: Trajectory = field(
        default_factory=Trajectory,
    )


def make_trajectory_wrapper_value():
    return AbstractValue.Make(TrajectoryWrapper())


def make_trajectory_value():
    return AbstractValue.Make(default_initial_trajectory)


class TrajectorySystem(LeafSystem):
    def __init__(
        self,
        plant: MultibodyPlant,
        update_rate: float = 1.0 / 1000.0,
        warmup_time: float = 1.0,
    ):
        super().__init__()
        self.plant = plant

        self.update_rate = update_rate
        self.index = 0

        self.warmup_time = warmup_time

        # Abstract States: Trajectories
        self.base_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.left_foot_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.right_foot_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.left_hand_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.right_hand_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.left_elbow_index = self.DeclareAbstractState(
            make_trajectory_value()
        )
        self.right_elbow_index = self.DeclareAbstractState(
            make_trajectory_value()
        )

        self.abstract_states = [
            self.base_index,
            self.left_foot_index,
            self.right_foot_index,
            self.left_hand_index,
            self.right_hand_index,
            self.left_elbow_index,
            self.right_elbow_index,
        ]

        self.trajectory_port = self.DeclareAbstractOutputPort(
            "trajectory",
            alloc=lambda: make_trajectory_wrapper_value(),
            calc=self.output_callback,
        ).get_index()

        # Declare Initialization Event: Update Trajectory Output
        def on_initialization(context, event):
            self.update(context, event)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        # Declare Periodic Event: Update Trajectory Output
        def on_periodic(context, event):
            if context.get_time() > self.warmup_time:
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
        wrapper = output.get_mutable_value()

        values = list(
            map(
                lambda x: (
                    context
                    .get_mutable_abstract_state(x)
                    .get_mutable_value()
                ),
                self.abstract_states,
            ),
        )

        loop_iterables = zip(
            fields(wrapper),
            values,
        )

        for field_name, value in loop_iterables:
            setattr(wrapper, field_name.name, value)

        output.set_value(wrapper)

    def update(self, context, event):
        position_target, rotation_target = self.desired_position(context)
        loop_iterables = zip(
            self.abstract_states,
            position_target,
            rotation_target,
        )
        for abstract_state, position, rotation in loop_iterables:
            state = context.get_mutable_abstract_state(
                abstract_state,
            )
            value = state.get_mutable_value()
            value['x'] = position[2]
            value['dx'] = position[1]
            value['ddx'] = position[0]
            value['w'] = rotation[2]
            value['dw'] = rotation[1]
            value['ddw'] = rotation[0]
            state.set_value(value)

    def desired_position(self, context):
        # Base Tracking:
        # Position:
        base_ddx = np.zeros((3,))
        base_dx = np.zeros_like(base_ddx)
        # Update Base Position based on average foot position:
        base_z = 0.8 + 0.2 * np.sin(self.index / 1000.0)
        base_x = np.array([0.046, 0.00, base_z])

        # Position and Velocity Tracking:
        # amplitude_scale = 0.3
        # frequency_scale = 2000.0
        # theta = amplitude_scale * np.sin(self.index / frequency_scale)
        # dtheta = amplitude_scale / frequency_scale * np.cos(self.index / frequency_scale)
        # ddtheta = -amplitude_scale / frequency_scale**2 * np.sin(self.index / frequency_scale)
        # base_w = np.array([1.0, 0.0, 0.0, theta])
        # base_w = base_w / np.linalg.norm(base_w)
        # base_dw = np.array([0.0, 0.0, dtheta])
        # base_ddw = np.array([0.0, 0.0, ddtheta])

        base_w = np.array([1.0, 0.0, 0.0, 0.0])
        base_dw = np.array([0.0, 0.0, 0.0])
        base_ddw = np.array([0.0, 0.0, 0.0])

        # Foot Tracking:
        # Position:
        foot_ddx = np.zeros_like(base_ddx)
        foot_dx = np.zeros_like(foot_ddx)
        left_foot_x = np.array([0.009, 0.100, 0.000])
        right_foot_x = np.array([0.009, -0.100, 0.000])
        # Rotation:
        foot_ddw = np.zeros_like(base_ddx)
        foot_dw = np.zeros_like(foot_ddw)
        left_foot_w = np.array([1.0, 0.0, 0.0, 0.0])
        right_foot_w = np.array([1.0, 0.0, 0.0, 0.0])

        # Hand Tracking:
        # Position:
        hand_ddx = np.zeros_like(base_ddx)
        hand_dx = np.zeros_like(hand_ddx)
        left_hand_x = np.array([0.19, 0.3, 0.92])
        right_hand_x = np.array([0.19, -0.3, 0.92])
        # Rotation:
        hand_ddw = np.zeros_like(base_ddx)
        hand_dw = np.zeros_like(hand_ddw)
        left_hand_w = np.array([1.0, 0.0, 0.0, 0.0])
        right_hand_w = np.array([1.0, 0.0, 0.0, 0.0])

        # Elbow Tracking:
        # Position:
        elbow_ddx = np.zeros_like(base_ddx)
        elbow_dx = np.zeros_like(elbow_ddx)
        elbow_z = 1.13 + 0.2 * np.sin(self.index / 1000.0)
        left_elbow_x = np.array([-0.11, 0.32, elbow_z])
        right_elbow_x = np.array([-0.11, -0.32, elbow_z])
        # Rotation:
        elbow_ddw = np.zeros_like(base_ddx)
        elbow_dw = np.zeros_like(elbow_ddw)
        left_elbow_w = np.array([1.0, 0.0, 0.0, 0.0])
        right_elbow_w = np.array([1.0, 0.0, 0.0, 0.0])

        self.index += 1

        position_target = [
            [base_ddx, base_dx, base_x],
            [foot_ddx, foot_dx, left_foot_x],
            [foot_ddx, foot_dx, right_foot_x],
            [hand_ddx, hand_dx, left_hand_x],
            [hand_ddx, hand_dx, right_hand_x],
            [elbow_ddx, elbow_dx, left_elbow_x],
            [elbow_ddx, elbow_dx, right_elbow_x],
        ]
        rotation_target = [
            [base_ddw, base_dw, base_w],
            [foot_ddw, foot_dw, left_foot_w],
            [foot_ddw, foot_dw, right_foot_w],
            [hand_ddw, hand_dw, left_hand_w],
            [hand_ddw, hand_dw, right_hand_w],
            [elbow_ddw, elbow_dw, left_elbow_w],
            [elbow_ddw, elbow_dw, right_elbow_w],
        ]

        return position_target, rotation_target
