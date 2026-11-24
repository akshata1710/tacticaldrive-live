# TacticalDrive Live — Real-Time Perception & Behavior Planning Prototype

Built for ADAS Behavior Planning.
Demonstrates a real-time-style perception pipeline (sensor noise simulation →
multi-object tracking → uncertainty-aware output) over a ROS2-shaped pub/sub
architecture, plus critical-scenario extraction from simulated fleet data.


## What's here (this phase: perception)

```
tacticaldrive-live/
├── bus/topic_bus.py          # ROS2-topic-equivalent pub/sub + QoS (see docs/ros2_port.md)
├── sim/ground_truth_sim.py   # multi-agent kinematic sim, publishes /sim/ground_truth
├── perception/perception_node.py
│   ├── SensorNoiseModel      # degrades GT into noisy/dropped/false-positive detections
│   ├── MultiObjectTracker    # Hungarian assignment + per-track Kalman filter (CV model)
│   └── PerceptionNode        # wires sim -> noise -> tracker -> /perception/tracked_objects
├── scripts/eval_tracking.py  # tracking metrics (RMSE, recall, ID switches) +
│                              # critical-scenario extraction via TTC
└── docs/ros2_port.md         # exact mapping to real rclpy for when you have a ROS2 env
```

## Run it

```bash
cd tacticaldrive-live
pip install numpy scipy filterpy --break-system-packages

PYTHONPATH=. python3 perception/perception_node.py   # smoke test, prints one frame
PYTHONPATH=. python3 scripts/eval_tracking.py         # full episode + metrics + scenario mining
```

## What each piece demonstrates for the JD

| JD requirement | Where it lives |
|---|---|
| Non-ML planning / classical methods understanding | `MultiObjectTracker` (Hungarian + Kalman are the classical, non-learned baseline — same story extends to the planner phase with FSM/behavior trees) |
| "Extract critical scenarios from fleet/sim data" | `scripts/eval_tracking.py::extract_critical_scenarios` — TTC-based flagging, run on **perception output**, not ground truth (matches how real scenario mining works) |
| Scalable/efficient data management systems | `topic_bus.py`'s QoS-bounded history buffers, decoupled node boundaries |
| Environment modeling / perception background | Full detect→associate→track→publish pipeline with propagated covariance |
| Real-time systems understanding | `deadline_ms` tracking, publish-rate discipline, `docs/ros2_port.md` mapping to actual DDS QoS |
| C++ / embedded porting | Planned next phase: export tracker inference-critical path or a trained model to a C++ node (see roadmap below) |

## Design decisions worth explaining out loud in an interview

- **Tracks carry covariance, not just a point estimate.** This is the direct
  link to the POMDP framing: planning downstream should treat agent state as
  a belief, not ground truth. Most toy tracking demos skip this.
- **Confirmed-track gating (`hits >= 2`).** Suppresses single-frame false
  positives from ever reaching the planner — a cheap, standard technique for
  trading a little latency for a lot of stability.
- **Scenario mining runs on perception output, not ground truth.** A
  real fleet-data pipeline never has ground truth to mine against; testing
  the extraction logic against noisy tracker output (and seeing it still work)
  is the actually-honest version of this deliverable.
- **Business logic is transport-agnostic.** `SensorNoiseModel` and
  `MultiObjectTracker` have zero dependency on `topic_bus` — they're pure
  functions/classes over dicts and numpy arrays, unit-testable without any
  pub/sub machinery, and portable to real rclpy without changes (see
  `docs/ros2_port.md`).

## Roadmap (next phases)

1. **Behavior planner node** — FSM/behavior-tree baseline subscribing to
   `/perception/tracked_objects`, publishing `/planning/decision`
2. **Imitation-learned planner** — behavior cloning on logged episodes,
   explicit distribution-shift failure demo, then DAgger fix
3. **Safety-cage wrapper** — rule-based override around the learned policy
   using TTC thresholds already built in `eval_tracking.py`
4. **C++ inference node** — export trained model to ONNX, C++ ROS2 node,
   benchmark Python vs. C++ p50/p99 inference latency
5. **Live dashboard** — FastAPI + WebSocket bridge off the topic bus,
   React front end showing live tracks + planner decision + latency graph
   (direct reuse of the CollabMind WebSocket connection-manager pattern)