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

from matplotlib import colors as mcolors
from matplotlib.backend_bases import TimerBase

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


_POINT_COLOR_CYCLE: list[tuple[float, float, float]] = [
    mcolors.to_rgb("#1f77b4"),
    mcolors.to_rgb("#ff7f0e"),
    mcolors.to_rgb("#2ca02c"),
    mcolors.to_rgb("#d62728"),
]


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
        self._point_base_colors: list[tuple[float, float, float]] = []
        self._plot_points_artist = None

        # Keep track of frames to show in the frame subplot.
        self._frames_from_points: list[tuple[np.ndarray, tuple[float, float, float]]] = []

        # Animation state.
        self._animation_timer: TimerBase | None = None
        self._animation_points: list[np.ndarray] = []
        self._animation_colors: list[tuple[float, float, float]] = []
        self._animation_index: int = 0
        self._animation_point_artist = None

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

        # Button to start the interpolation animation.
        animate_ax = self.fig.add_axes([0.62, 0.05, 0.18, 0.06])
        self._animate_button = Button(animate_ax, "Animate path")
        self._animate_button.on_clicked(self._on_animate_clicked)

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

        self._stop_animation()

        color_index = len(self._points) % len(_POINT_COLOR_CYCLE)
        base_color = _POINT_COLOR_CYCLE[color_index]

        self._points.append(point)
        self._point_base_colors.append(base_color)
        self._frames_from_points.append((_so3_exp(point), base_color))
        self._update_points()
        self._draw_frames()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _update_points(self) -> None:
        """Refresh the scatter artist that displays stored points on the right."""

        if self._plot_points_artist is not None:
            self._plot_points_artist.remove()
            self._plot_points_artist = None

        if self._points:
            points = np.vstack(self._points)
            xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
            self._plot_points_artist = self.ax_plot.scatter(
                xs,
                ys,
                zs,
                color=self._point_base_colors,
                s=40,
                depthshade=False,
            )

    def _draw_frames(self) -> None:
        """Draw the frames created from stored points."""

        self._reset_frame_axis()

        for R, base_color in self._frames_from_points:
            self._draw_single_frame(R, base_color)

    def _axis_colors_for_base(
        self, base_color: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], ...]:
        """Return RGB triples for the axes derived from ``base_color``."""

        return (base_color, base_color, base_color)

    def _draw_single_frame(
        self,
        R: np.ndarray,
        base_color: tuple[float, float, float],
        *,
        annotate: bool = True,
    ) -> None:
        """Draw a single frame using ``R`` and ``base_color``."""

        origin = np.asarray(self.chart(R), dtype=float)
        if origin.shape != (3,):  # pragma: no cover - simple input validation
            raise ValueError("chart(R) must return a 3-vector")

        axis_colors = self._axis_colors_for_base(base_color)

        for axis_index in range(3):
            direction = R[:, axis_index]
            self.ax_frame.quiver(
                *origin,
                *direction,
                color=axis_colors[axis_index],
                linewidth=2,
            )

            if annotate:
                tip = origin + direction
                self.ax_frame.text(
                    *tip,
                    str(axis_index + 1),
                    color=axis_colors[axis_index],
                    fontsize=10,
                    ha="center",
                    va="center",
                )

    def _on_animate_clicked(self, _event) -> None:
        """Handle clicks on the *Animate path* button."""

        self.start_animation()

    def start_animation(self) -> None:
        """Start animating an interpolated point if possible."""

        if len(self._points) < 2:
            return

        self._stop_animation()
        self._prepare_animation_path()
        if not self._animation_points:
            return

        self._animation_index = 0
        self._ensure_animation_timer()
        self._animation_timer.start()

    def _prepare_animation_path(self) -> None:
        """Compute the interpolation path through stored points."""

        path: list[np.ndarray] = []
        colors: list[tuple[float, float, float]] = []

        for idx in range(len(self._points) - 1):
            start = self._points[idx]
            end = self._points[idx + 1]
            start_color = self._point_base_colors[idx]
            end_color = self._point_base_colors[idx + 1]

            for step_index, alpha in enumerate(np.linspace(0.0, 1.0, 11)):
                if idx > 0 and step_index == 0:
                    continue

                point = (1.0 - alpha) * start + alpha * end
                color = start_color if alpha < 1.0 else end_color
                path.append(point)
                colors.append(color)

        self._animation_points = path
        self._animation_colors = colors

    def _ensure_animation_timer(self) -> None:
        """Create and configure the Matplotlib timer for animations."""

        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

        timer = self.fig.canvas.new_timer(interval=200)
        timer.add_callback(self._advance_animation)
        if hasattr(timer, "single_shot"):
            timer.single_shot = False
        self._animation_timer = timer

    def _advance_animation(self) -> None:
        """Advance the interpolation animation by one step."""

        if self._animation_index >= len(self._animation_points):
            self._stop_animation()
            self.fig.canvas.draw_idle()
            return

        point = self._animation_points[self._animation_index]
        color = self._animation_colors[self._animation_index]

        self._draw_frames()
        self._draw_single_frame(_so3_exp(point), color)
        self._update_animation_point_artist(point, color)

        self._animation_index += 1
        if self._animation_index >= len(self._animation_points):
            self._stop_animation(clear_artists=False)

        self.fig.canvas.draw_idle()

    def _update_animation_point_artist(
        self, point: np.ndarray, color: tuple[float, float, float]
    ) -> None:
        """Draw or update the animated point on the right subplot."""

        if self._animation_point_artist is not None:
            self._animation_point_artist.remove()
            self._animation_point_artist = None

        self._animation_point_artist = self.ax_plot.scatter(
            [point[0]],
            [point[1]],
            [point[2]],
            color=[color],
            s=70,
            depthshade=False,
            marker="o",
        )

    def _stop_animation(self, *, clear_artists: bool = True) -> None:
        """Stop any running animation and optionally clear artists."""

        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

        self._animation_points = []
        self._animation_colors = []
        self._animation_index = 0

        if clear_artists:
            if self._animation_point_artist is not None:
                self._animation_point_artist.remove()
                self._animation_point_artist = None

            self._draw_frames()

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
    visualizer.show()
