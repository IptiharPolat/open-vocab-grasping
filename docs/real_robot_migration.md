# Migration to a real RGB-D camera and SO-ARM101

The perception and ranking contracts can be retained, but simulator-specific
observation, kinematics and safety code must be replaced. This is engineering
work, not a configuration-only switch.

## 1. Real RGB-D observation adapter

Implement a `CameraSource` interface returning synchronized RGB, depth in metres,
`K`, distortion coefficients, timestamps and `T_base_camera`. A RealSense-style
adapter should align depth to colour, reject invalid/saturated depth, expose the
device depth scale and deproject only after distortion handling. Save raw frames
and calibration metadata so simulation and hardware episodes share the same
artifact schema.

Acceptance checks:

- measure several known distances with a board and keep median depth error within
  a chosen sensor-specific tolerance;
- project a checkerboard/AprilTag corner through `K` and confirm pixel residuals;
- verify table points form a plane at the measured base-frame height;
- quantify RGB/depth temporal skew while moving an object.

## 2. Camera-to-robot calibration

For a fixed external camera, estimate `T_base_camera` with an eye-to-hand
calibration using a board rigidly attached to the end effector. Record at least
15 diverse robot poses, reject outliers, and validate on poses excluded from the
fit. Store the calibration with camera serial number, resolution, robot identity,
timestamp and residuals. The existing transform chain then becomes:

```text
T_base_grasp = T_base_camera @ T_camera_grasp @ T_grasp_tool
```

Never tune axis signs until one pose “looks right”; render base/camera/tool axes
and validate independent point correspondences.

## 3. SO-ARM101 robot adapter

Replace Panda-specific link indices, joint limits and PyBullet IK with an
`ArmController` implementation for SO-ARM101. Required methods are home, current
joint state, FK, bounded IK, collision-checked trajectory execution, gripper
width command and emergency stop. Obtain or build a URDF whose joint zero,
direction and link dimensions match the physical arm, then measure
`T_wrist_tool` and the gripper TCP rather than copying Panda offsets.

Because SO-ARM101 uses servo position control, add:

- joint-position, velocity, current/torque and temperature limits;
- a conservative Cartesian workspace and table exclusion volume;
- time-parameterized waypoints with feedback timeout and tracking-error abort;
- an open/closed gripper calibration mapping command units to jaw width;
- a physical E-stop and software dead-man path outside the perception process.

The LeRobot transport can implement the low-level controller, while IK and
collision checking can use a validated URDF model through Pinocchio, IKPy or a
small MoveIt/ROS 2 bridge. This project should consume the controller interface,
not issue raw servo writes from detector output.

## 4. Real success measurement

Simulator body pose is unavailable. Use at least two independent signals:

- gripper closure/current indicates an object between the fingers rather than a
  fully closed empty grasp;
- post-lift RGB-D tracking shows the selected object/mask moved upward with the
  tool and remains present after a short hold.

For dataset-quality evaluation, also record a human success label and keep it
separate from the automatic heuristic.

## 5. Safe rollout sequence

1. Offline playback: run detector, point cloud and grasp ranking on recorded
   frames with all motor output disabled.
2. Shadow mode: compare planned TCP poses with measured reachable workspace.
3. Empty-table motion at low speed/current; validate TCP and table clearance.
4. Soft-object grasps with a human holding the E-stop.
5. Fixed objects, then randomized scenes; only afterward enable batch trials.

Before hardware execution, add integration tests for calibration versioning,
stale-frame rejection, joint-limit clipping, communication loss, E-stop and
planner/controller unit conventions. YOLO domain adaptation and a GPU-backed
GraspNet or another modern grasp proposal model can then be evaluated without
changing these safety boundaries.
