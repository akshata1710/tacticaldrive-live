# Porting `topic_bus.py` to real ROS2 (rclpy)

This project's `bus/topic_bus.py` deliberately mirrors ROS2's node/topic/QoS
model so the architecture ports with minimal rewrite. This doc is the mapping
you'd walk an interviewer through.

## Concept mapping

| This project                          | Real ROS2 equivalent                                   |
|----------------------------------------|------------------------------------------------------------|
| `bus.get_topic(name, qos)`            | `node.create_publisher(MsgType, name, qos_profile)`      |
| `topic.subscribe(callback)`           | `node.create_subscription(MsgType, name, callback, qos)` |
| `QoSProfile(reliability=...)`         | `rclpy.qos.QoSProfile(reliability=ReliabilityPolicy...)` |
| `QoSProfile(deadline_ms=...)`         | `QoSProfile(deadline=Duration(...))`                     |
| in-process dict message               | a `.msg` file (e.g. `TrackedObjectArray.msg`)             |
| synchronous callback dispatch          | `rclpy.spin()` executor scheduling (real threading/timing) |
| module-level `bus` singleton           | the DDS domain (processes discover each other automatically) |

## What changes when you port `PerceptionNode`

```python
# Today (this repo):
class PerceptionNode:
    def __init__(self, rate_hz=20.0, ego_id=0):
        ...
        self.out_topic = bus.get_topic("/perception/tracked_objects", qos=...)
        bus.get_topic("/sim/ground_truth").subscribe(self._on_ground_truth)

# Real ROS2 (rclpy):
class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5,
                          deadline=Duration(seconds=0.15))
        self.publisher_ = self.create_publisher(TrackedObjectArray,
                                                  '/perception/tracked_objects', qos)
        self.subscription = self.create_subscription(
            AgentStateArray, '/sim/ground_truth', self._on_ground_truth, qos)

    def _on_ground_truth(self, msg):
        ...  # identical tracker logic (SensorNoiseModel, MultiObjectTracker unchanged)
        self.publisher_.publish(tracked_msg)
```

**The tracker logic itself (`SensorNoiseModel`, `Track`, `MultiObjectTracker`)
does not change at all** -- it's pure Python/numpy with no bus dependency.
Only the node's `__init__` and I/O plumbing change. This separation
(business logic vs. transport) is itself a software-design point worth
making in the interview: the Kalman/Hungarian tracking core is unit-testable
in isolation from ROS2 entirely, which is exactly how you'd want it
structured for CI (you can't easily run `colcon test` in most CI runners
without a full ROS2 image, but you can always `pytest` the tracker).

## Why deadline_ms matters here specifically

`/perception/tracked_objects` publishing late means the planner is reasoning
about a stale world -- directly the "firm real-time" failure mode from the
theory doc. Setting `deadline_ms` on the topic and tracking `deadline_misses`
gives you a concrete, demoable number ("perception missed its publish
deadline in 0.2% of frames under simulated sensor dropout") instead of a
hand-wavy "it's fast enough" claim.

## Next real ROS2 steps (when running outside this sandbox)

1. `sudo apt install ros-humble-desktop` (or use the official Docker image)
2. `ros2 pkg create --build-type ament_python tacticaldrive_perception`
3. Define `.msg` files for `AgentState`, `AgentStateArray`, `TrackedObject`,
   `TrackedObjectArray`
4. Copy `SensorNoiseModel` / `Track` / `MultiObjectTracker` into the new
   package unchanged; rewrite only `PerceptionNode.__init__` per the mapping above
5. `ros2 topic hz /perception/tracked_objects` to verify real publish rate
   under load -- this is the live demo you can show in an interview
