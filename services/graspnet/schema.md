# GraspNet file interface schema v1.0

The main environment writes a compressed NPZ request:

| Key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version` | Unicode | scalar | `1.0` |
| `rgb` | uint8 | H x W x 3 | RGB image |
| `depth_m` | float32 | H x W | optical-axis depth in metres |
| `intrinsic` | float64 | 3 x 3 | OpenCV pinhole K |
| `workspace_mask` | bool | H x W | valid inference workspace |

The isolated service returns a compressed NPZ response using the official
GraspNet camera-frame convention, with lengths in metres. `centers` are network
reference translations; they are not yet Panda TCP centers:

| Key | dtype | shape |
| --- | --- | --- |
| `centers` | float32 | N x 3 |
| `rotations` | float32 | N x 3 x 3 |
| `widths` | float32 | N |
| `depths` | float32 | N |
| `scores` | float32 | N |
| `collision` | bool | N |

Optional scalar metadata includes `schema_version=1.0` and `generator`. The
official service sets GraspNet values; `geometric_infer.py` writes
`geometric_protocol_baseline_not_graspnet` and is only a CPU transport test.

For execution, the main environment constructs the Panda TCP pose as follows:

```text
p_camera_tcp = p_camera_reference + R_camera_grasp[:, 0] * grasp_depth
R_grasp_tool = [[ 0, 0, 1],
                [ 0, 1, 0],
                [-1, 0, 0]]
T_world_tool = T_world_camera @ T_camera_grasp @ T_grasp_tool
```

GraspNet +x is approach and +y is closing. Panda tool +z is approach and its
URDF fingers translate along tool +/-y, hence the fixed right-handed mapping.
