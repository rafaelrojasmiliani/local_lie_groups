"""Visualization tools for local Lie group examples.

This module defines a small utility class that bundles all Matplotlib
initialisation required for the examples in this repository.  The class creates
one window that is split into two vertical plots:

* the **left** subplot shows a coordinate frame drawn from an orthogonal
  matrix.  The location of this frame is obtained by mapping the matrix through
  a chart :math:`f: \mathrm{SO}(3) \to \mathbb{R}^3` supplied by the user;
* the **right** subplot is a general purpose three‑dimensional plot which can
  be used by examples to display additional data.

The intention is to make it easy to visualise how different orientations are
represented in a given chart.
"""

from __future__ import annotations

from typing import Callable

import matplotlib

# Use a non‑interactive backend so that unit tests or headless environments do
# not require a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


class FrameVisualizer:
    """Visualise coordinate frames using Matplotlib.

    Parameters
    ----------
    chart:
        Callable implementing a chart :math:`f: \mathrm{SO}(3) \to \mathbb{R}^3`.
        For an orthogonal matrix ``R`` this function returns the coordinates
        where the frame should be drawn in the left subplot.
    figsize:
        Optional Matplotlib figure size.  Defaults to ``(10, 5)``.
    """

    def __init__(self, chart: Callable[[np.ndarray], np.ndarray], figsize=(10, 5)):
        self.chart = chart
        self.fig = plt.figure(figsize=figsize)

        # Left subplot for the oriented frame in chart coordinates.
        self.ax_frame = self.fig.add_subplot(1, 2, 1, projection="3d")
        # Right subplot for general 3‑D visualisations.
        self.ax_plot = self.fig.add_subplot(1, 2, 2, projection="3d")

        self._configure_axes()

    # ------------------------------------------------------------------
    def _configure_axes(self) -> None:
        """Apply a basic configuration to both subplots."""

        for ax in (self.ax_frame, self.ax_plot):
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            # Equal aspect ratio for a more faithful representation.
            ax.set_box_aspect([1, 1, 1])

        self.ax_frame.set_title("Frame in chart")
        self.ax_plot.set_title("3D plot")

    # ------------------------------------------------------------------
    def draw_frame(self, R: np.ndarray, length: float = 1.0) -> None:
        """Draw an oriented frame given by ``R`` in the left subplot.

        Parameters
        ----------
        R:
            Orthogonal ``3 x 3`` matrix representing a rotation.
        length:
            Length of the frame's axes.  Defaults to ``1.0``.
        """

        if R.shape != (3, 3):  # pragma: no cover - simple input validation
            raise ValueError("R must be a 3x3 matrix")

        # Clear previous content while keeping axis limits and titles.
        self.ax_frame.cla()
        self.ax_frame.set_title("Frame in chart")
        self.ax_frame.set_xlim([-1, 1])
        self.ax_frame.set_ylim([-1, 1])
        self.ax_frame.set_zlim([-1, 1])
        self.ax_frame.set_box_aspect([1, 1, 1])

        origin = np.asarray(self.chart(R), dtype=float)
        if origin.shape != (3,):  # pragma: no cover - simple input validation
            raise ValueError("chart(R) must return a 3‑vector")

        # Draw the axes of the frame.  Columns of R correspond to the directions
        # of the x, y and z axes of the rotated frame.
        self.ax_frame.quiver(*origin, *R[:, 0] * length, color="r", linewidth=2)
        self.ax_frame.quiver(*origin, *R[:, 1] * length, color="g", linewidth=2)
        self.ax_frame.quiver(*origin, *R[:, 2] * length, color="b", linewidth=2)

        # Update the figure without blocking to allow successive calls.
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def show(self) -> None:
        """Display the Matplotlib figure."""

        plt.tight_layout()
        plt.show()


def show_example() -> None:
    """Simple example demonstrating :class:`FrameVisualizer`.

    The example uses the identity chart, i.e. it draws the frame at the
    origin, and displays the standard basis.
    """

    chart = lambda R: np.zeros(3)
    visualizer = FrameVisualizer(chart)
    visualizer.draw_frame(np.eye(3))
    visualizer.show()

