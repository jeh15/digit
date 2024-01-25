from dataclasses import dataclass

import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import (
    LeafSystem,
    Context,
)


@dataclass
class ContextWrapper:
    context: Context


def make_context_wrapper_value(plant):
    return AbstractValue.Make(ContextWrapper(plant.CreateDefaultContext()))


# Example Producer (Output Port)
class PlantContextSystem(LeafSystem):
    def __init__(self, plant):
        super().__init__()
        nx = plant.num_positions() + plant.num_velocities()
        self.plant = plant
        self.plant_state_port = self.DeclareVectorInputPort("plant_state", nx)

        def needs_update(plant_context, x):
            context_x = plant.GetPositionsAndVelocities(plant_context)
            return np.any(context_x != x)

        def calc_context(context, output):
            x = self.plant_state_port.Eval(context)
            wrapper = output.get_mutable_value()
            plant_context = wrapper.context
            if needs_update(plant_context, x):
                plant.SetPositionsAndVelocities(plant_context, x)

        self.plant_context_port = self.DeclareAbstractOutputPort(
            "plant_context",
            alloc=lambda: make_context_wrapper_value(plant),
            calc=calc_context,
        ).get_index()
