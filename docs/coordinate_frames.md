# Coordinate-frame conventions

All lengths are metres, angles radians, rotations right-handed, and homogeneous
column vectors are used. A transform is named `T_destination_source` and maps
`p_destination = T_destination_source @ p_source`.

- `world`: PyBullet world; +x away from the Panda base, +y left across the table,
  +z up. In stage 1 the Panda base coincides with world, so `T_base_world = I`.
- `camera`: OpenCV optical frame; +x image-right, +y image-down, +z forward.
- `tool`: Panda `panda_grasptarget`; +z is approach and the URDF prismatic
  finger joints move along +/-y, so +y is the closing/opening axis.
- `grasp`: the geometric baseline already uses the Panda tool convention. The
  official GraspNet frame uses +x as approach, +y as closing and +z as gripper
  height.

The fixed camera pose is built analytically. `forward = normalize(target-eye)`,
`right = normalize(forward x up_world)`, and `down = -(right x forward)`, so
`R_world_camera = [right, down, forward]`. No unexplained axis swap is used.

Projection is the standard pinhole equation `u=fx*x/z+cx`, `v=fy*y/z+cy`.
PyBullet's OpenGL depth buffer `d` is converted to optical-axis depth by
`z = far*near / (far - (far-near)*d)`.

The GraspNet adapter chain is explicitly:

`T_base_grasp = T_base_world @ T_world_camera @ T_camera_grasp @ T_grasp_tool`.

The verified fixed GraspNet rotation adapter is:

```text
R_grasp_tool = [[ 0, 0, 1],
                [ 0, 1, 0],
                [-1, 0, 0]]
```

Thus Panda tool +z is GraspNet +x, Panda tool +y is GraspNet +y, and Panda tool
+x is -GraspNet +z. The official reference translation is also advanced by the
per-candidate depth along GraspNet +x before it becomes the Panda contact TCP:
`p_camera_tcp = p_camera_reference + R_camera_grasp[:,0] * depth`.

For the geometric baseline, candidate centers target link 11 directly:
the Panda URDF already defines `panda_grasptarget` 0.105 m along the hand axis,
so applying another 0.105 m translation would double-count the TCP offset.

For the perception-quality mug, rendered geometry includes the handle while
collision geometry is a cylindrical proxy. Its geometric grasp center is fitted
from segmented RGB-D points with deterministic circle RANSAC, rejecting handle
outliers; no simulator body pose is used to generate the candidate.

The open-vocabulary scene generator does not receive that segmented target point
set. It generates candidates from all non-table workspace clusters, transforms
their centers with `T_camera_world = inverse(T_world_camera)`, and applies the
text box/depth association before any truth-only metric is evaluated.
