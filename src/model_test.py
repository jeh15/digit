import os
from absl import app

from pydrake.geometry import (
    StartMeshcat,
)
from pydrake.visualization import ModelVisualizer


def main(argv=None):
    # Start meshcat server:
    meshcat = StartMeshcat()

    # Load XML file:
    xml_path = "models/digit.urdf"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        xml_path,
    )

    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().AddModels(filepath)
    visualizer.Run()


if __name__ == "__main__":
    app.run(main)
