# Visualising $\mathrm{SO}(3)$ with the Exponential Chart

This project provides a small, interactive visualiser for the Lie group
$\mathrm{SO}(3)$, the group of 3D rotations.  It is designed to build intuition
for how orientations are represented in robotics, computer graphics, and
navigation by stepping through the sequence of ideas that makes Lie methods so
powerful: from a practical motivation, to the associated Lie algebra, and
finally to the exponential chart used by the tool.

## Why Lie groups for orientations?

Representing orientations in three-dimensional space is a fundamental problem.
Euler angles are compact and intuitive but suffer from gimbal lock.  Rotation
matrices are robust but involve redundant parameters and constraints.  Quaternions
avoid singularities but introduce unit-norm constraints that can be tricky to
handle.  Lie groups provide a unifying framework: $\mathrm{SO}(3)$ forms a smooth
manifold equipped with a group operation, and working directly on this manifold
lets us interpolate, differentiate, and optimise rotations without leaving the
space of valid orientations.

## The Lie algebra $\mathfrak{so}(3)$

Every matrix Lie group has an associated Lie algebra, the tangent space at the
identity.  For $\mathrm{SO}(3)$ this is the set of $3 \times 3$ skew-symmetric
matrices.  A vector $\omega = [\omega_x,\omega_y,\omega_z]^\top$ in
$\mathbb{R}^3$ can be mapped into the algebra via the hat operator,
\(\hat{\omega}\), producing the matrix

\[
\hat{\omega} = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}.
\]

The Lie bracket in $\mathfrak{so}(3)$ coincides with the cross product in
$\mathbb{R}^3$, revealing the close relationship between angular velocities and
infinitesimal rotations.  Working in the algebra allows us to reason about small
motions in a vector space while preserving the geometric structure of
$\mathrm{SO}(3)$.

## The exponential chart

To move from the algebra back to the group we use the matrix exponential.  The
exponential map, $\exp\colon \mathfrak{so}(3) \to \mathrm{SO}(3)$, takes a
skew-symmetric matrix and returns a rotation matrix.  When we parameterise the
algebra with the vector $\omega$, the chart used in this project is

\[
\chi(\omega) = \exp(\hat{\omega}) \in \mathrm{SO}(3).
\]

This chart is valid in a neighbourhood of the identity and, thanks to
Rodrigues' rotation formula, has a closed-form expression that is efficient and
numerically stable.  The visualiser animates how points in algebra coordinates
map to orientations on the sphere using this exponential chart.

## Project structure

- `local_lie_group/visualization.py` contains the Matplotlib-based
  `FrameVisualizer`, which renders orientation frames and plots additional data
  side by side.
- `main.py` launches a ready-to-run example that showcases the exponential
  chart.

## Getting started

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install the minimal dependencies:

   ```bash
   pip install matplotlib numpy
   ```

3. Run the example script:

   ```bash
   python -m main
   ```

   An interactive window appears with sliders controlling the Lie algebra
   coordinates $(\omega_x, \omega_y, \omega_z)$.  Adjust them to explore how the
   exponential chart maps algebra vectors to orientations.

## Extending the visualiser

You can adapt the `FrameVisualizer` to experiment with alternative charts or to
superimpose trajectories.  Supply a custom chart function
`f: \mathrm{SO}(3) \to \mathbb{R}^3` when constructing the class and use the
public methods to add frames and points to the plots.  The implementation in
`local_lie_group/visualization.py` serves as a reference for how to define the
hat operator, compute the matrix exponential, and orchestrate animations.

## Further reading

- F. Bullo and A. Lewis, *Geometric Control of Mechanical Systems*, Springer.
- E. Sola, J. Deray, and D. Atcheson, "A micro Lie theory for state estimation in
  robotics," 2018.
- R. Murray, Z. Li, and S. Sastry, *A Mathematical Introduction to Robotic
  Manipulation*, CRC Press.

These resources provide deeper insight into Lie groups, their algebras, and the
exponential map in the context of robotics and control.
