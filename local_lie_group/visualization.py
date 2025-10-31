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
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

import matplotlib

# Use a non‑interactive backend so that unit tests or headless environments do
# not require a display server.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
from matplotlib.widgets import Button, Slider


def _hat(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix associated with ``vector``."""

    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    """Compute ``exp(S(vector))`` using Rodrigues' rotation formula."""

    theta = float(np.linalg.norm(vector))
    K = _hat(vector)

    if theta < 1e-12:
        return np.eye(3)

    K2 = K @ K
    if theta < 1e-4:
        sin_over_theta = 1.0 - (theta ** 2) / 6.0
        one_minus_cos_over_theta2 = 0.5 - (theta ** 2) / 24.0
    else:
        sin_over_theta = np.sin(theta) / theta
        one_minus_cos_over_theta2 = (1.0 - np.cos(theta)) / (theta * theta)

    return np.eye(3) + sin_over_theta * K + one_minus_cos_over_theta2 * K2


class FrameVisualizer:
    r"""Visualise coordinate frames using Matplotlib.

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

        # Reserve some space at the bottom for the interactive controls that
        # allow users to choose a point in the chart coordinates.
        self.fig.subplots_adjust(bottom=0.25)

        # Left subplot for the oriented frame in chart coordinates.
        self.ax_frame = self.fig.add_subplot(1, 2, 1, projection="3d")
        # Right subplot for general 3‑D visualisations.
        self.ax_plot = self.fig.add_subplot(1, 2, 2, projection="3d")

        # Storage for points that the user adds through the controls.
        self._points: list[np.ndarray] = []
        self._plot_points_artist = None

        # Keep track of frames to show in the frame subplot.
        self._frames_from_points: list[np.ndarray] = []
        self._base_frame: tuple[np.ndarray, float] | None = None

        self._configure_axes()
        self._create_controls()

    # ------------------------------------------------------------------
    def _configure_axes(self) -> None:
        """Apply a basic configuration to both subplots."""

        self._reset_frame_axis()

        self.ax_plot.set_xlim([-3.5, 3.5])
        self.ax_plot.set_ylim([-3.5, 3.5])
        self.ax_plot.set_zlim([-3.5, 3.5])
        self.ax_plot.set_box_aspect([1, 1, 1])
        self.ax_plot.set_title("3D plot")

    # ------------------------------------------------------------------
    def _reset_frame_axis(self) -> None:
        """Clear and configure the frame axis for drawing frames."""

        self.ax_frame.cla()
        self.ax_frame.set_title("Frame in chart")
        self.ax_frame.set_xlim([-1, 1])
        self.ax_frame.set_ylim([-1, 1])
        self.ax_frame.set_zlim([-1, 1])
        self.ax_frame.set_box_aspect([1, 1, 1])

    # ------------------------------------------------------------------
    def _create_controls(self) -> None:
        """Create sliders and a button to add points to the plots."""

        # Sliders for the x, y and z coordinates with 0.1 resolution.
        slider_positions = {
            "x": [0.12, 0.13, 0.23, 0.03],
            "y": [0.39, 0.13, 0.23, 0.03],
            "z": [0.66, 0.13, 0.23, 0.03],
        }

        self._sliders = {
            name: Slider(
                self.fig.add_axes(position),
                label=f"{name.upper()}",
                valmin=-1.0,
                valmax=1.0,
                valinit=0.0,
                valstep=0.1,
            )
            for name, position in slider_positions.items()
        }

        # Button to add the selected point to both plots.
        button_ax = self.fig.add_axes([0.45, 0.05, 0.12, 0.06])
        self._add_point_button = Button(button_ax, "Add point")
        self._add_point_button.on_clicked(self._on_add_point_clicked)

    # ------------------------------------------------------------------
    def _on_add_point_clicked(self, _event) -> None:
        """Handle clicks on the *Add point* button."""

        self.add_point(
            np.array(
                [
                    float(self._sliders["x"].val),
                    float(self._sliders["y"].val),
                    float(self._sliders["z"].val),
                ],
                dtype=float,
            )
        )

    # ------------------------------------------------------------------
    def add_point(self, point: np.ndarray) -> None:
        """Add a point to the right subplot and its frame to the left.

        Parameters
        ----------
        point:
            ``(3,)`` array containing the coordinates of the point in chart
            space.
        """

        point = np.asarray(point, dtype=float)
        if point.shape != (3,):  # pragma: no cover - simple input validation
            raise ValueError("point must be a 3-vector")

        self._points.append(point)
        self._frames_from_points.append(_so3_exp(point))
        self._update_points()
        self._draw_frames()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _update_points(self) -> None:
        """Refresh the scatter artist that displays stored points on the right."""

        if self._points:
            points = np.vstack(self._points)
            xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        else:
            xs = ys = zs = np.array([])

        # Update the scatter on the right subplot (generic view).
        if self._plot_points_artist is None and self._points:
            self._plot_points_artist = self.ax_plot.scatter(
                xs, ys, zs, color="k", s=40, depthshade=False
            )
        elif self._plot_points_artist is not None:
            if self._points:
                self._plot_points_artist._offsets3d = (xs, ys, zs)
            else:  # pragma: no cover - no points to show
                self._plot_points_artist.remove()
                self._plot_points_artist = None

    def _draw_frames(self) -> None:
        """Draw the base frame and all frames created from stored points."""

        self._reset_frame_axis()

        frames: list[tuple[np.ndarray, float]] = []
        if self._base_frame is not None:
            frames.append(self._base_frame)
        frames.extend((R, 1.0) for R in self._frames_from_points)

        for R, length in frames:
            origin = np.asarray(self.chart(R), dtype=float)
            if origin.shape != (3,):  # pragma: no cover - simple input validation
                raise ValueError("chart(R) must return a 3-vector")

            self.ax_frame.quiver(
                *origin,
                *R[:, 0] * length,
                color="r",
                linewidth=2,
            )
            self.ax_frame.quiver(
                *origin,
                *R[:, 1] * length,
                color="g",
                linewidth=2,
            )
            self.ax_frame.quiver(
                *origin,
                *R[:, 2] * length,
                color="b",
                linewidth=2,
            )

    def draw_frame(self, R: np.ndarray, length: float = 1.0) -> None:
        """Draw an oriented frame given by ``R`` in the left subplot."""

        if R.shape != (3, 3):  # pragma: no cover - simple input validation
            raise ValueError("R must be a 3x3 matrix")

        self._base_frame = (np.asarray(R, dtype=float), float(length))
        self._draw_frames()

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

    def chart(R): return np.zeros(3)
    visualizer = FrameVisualizer(chart)
    visualizer.draw_frame(np.eye(3))
    visualizer.show()
