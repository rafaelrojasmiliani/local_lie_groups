"""Visualization utilities for local Lie group examples."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


def show_example():
    """Display a coordinate frame and a 3D point side by side."""
    fig = plt.figure(figsize=(10, 5))

    # Left: coordinate frame
    ax_frame = fig.add_subplot(1, 2, 1, projection="3d")
    length = 1.0
    ax_frame.quiver(0, 0, 0, length, 0, 0, color="r", linewidth=2)
    ax_frame.quiver(0, 0, 0, 0, length, 0, color="g", linewidth=2)
    ax_frame.quiver(0, 0, 0, 0, 0, length, color="b", linewidth=2)
    ax_frame.set_xlim([-1, 1])
    ax_frame.set_ylim([-1, 1])
    ax_frame.set_zlim([-1, 1])
    ax_frame.set_title("Coordinate Frame")

    # Right: 3D point
    ax_point = fig.add_subplot(1, 2, 2, projection="3d")
    ax_point.scatter([0.5], [0.5], [0.5], color="k", s=50)
    ax_point.set_xlim([-1, 1])
    ax_point.set_ylim([-1, 1])
    ax_point.set_zlim([-1, 1])
    ax_point.set_title("3D Point")

    plt.tight_layout()
    plt.show()
