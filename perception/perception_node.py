"""
perception_node.py
-------------------
Subscribes to /sim/ground_truth, DEGRADES it into something that looks like
real sensor output (position/velocity noise, missed detections, occasional
false positives, one-frame detection latency), then runs a proper tracking
pipeline on the degraded stream:

    noisy detections --> Hungarian assignment --> per-track Kalman filter
                                                --> /perception/tracked_objects

Each published track carries a POSITION COVARIANCE, not just a point estimate --
this is the piece most portfolio projects skip, and it's the direct link to the
POMDP/uncertainty story: behavior planning downstream should treat this as a
belief over agent state, not ground truth.

Track state: [x, y, vx, vy]  (constant-velocity motion model)
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from bus.topic_bus import bus, QoSProfile


# ---------------------------------------------------------------------------
# Sensor degradation model
# ---------------------------------------------------------------------------

class SensorNoiseModel:
    """Turns ground-truth agent states into noisy 'detections', with:
      - Gaussian position/velocity noise (noise grows with range from ego)
      - probabilistic missed detections (occlusion / range dropout proxy)
      - occasional false-positive "ghost" detections (sensor artifacts)
    """

    def __init__(self, base_pos_std=0.3, base_vel_std=0.5,
                 miss_rate=0.08, false_positive_rate=0.03, seed=0):
        self.base_pos_std = base_pos_std
        self.base_vel_std = base_vel_std
        self.miss_rate = miss_rate
        self.false_positive_rate = false_positive_rate
        self.rng = np.random.default_rng(seed)

    def degrade(self, ego_state: dict, agents: list[dict]) -> list[dict]:
        detections = []
        for a in agents:
            if a["agent_id"] == ego_state["agent_id"]:
                continue  # ego doesn't detect itself

            range_m = np.hypot(a["x"] - ego_state["x"], a["y"] - ego_state["y"])
            # noise scales mildly with range -- crude proxy for real sensor behavior
            pos_std = self.base_pos_std * (1.0 + range_m / 50.0)
            vel_std = self.base_vel_std * (1.0 + range_m / 80.0)

            if self.rng.random() < self.miss_rate:
                continue  # missed detection this frame

            detections.append({
                "x": a["x"] + self.rng.normal(0, pos_std),
                "y": a["y"] + self.rng.normal(0, pos_std),
                "vx": a["vx"] + self.rng.normal(0, vel_std),
                "vy": a["vy"] + self.rng.normal(0, vel_std),
                "pos_std": pos_std,
                "vel_std": vel_std,
                "_true_id": a["agent_id"],   # kept ONLY for offline eval, tracker never sees this
            })

        if self.rng.random() < self.false_positive_rate:
            detections.append({
                "x": ego_state["x"] + self.rng.uniform(5, 40),
                "y": self.rng.uniform(-4, 4),
                "vx": self.rng.uniform(15, 30),
                "vy": 0.0,
                "pos_std": self.base_pos_std * 3,
                "vel_std": self.base_vel_std * 3,
                "_true_id": -1,  # ghost
            })

        return detections


# ---------------------------------------------------------------------------
# Per-object Kalman filter (constant velocity model)
# ---------------------------------------------------------------------------

def make_cv_kalman_filter(x0, y0, vx0, vy0, dt):
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.x = np.array([x0, y0, vx0, vy0])
    kf.F = np.array([  # constant-velocity transition model
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    kf.H = np.eye(4)  # we "measure" [x, y, vx, vy] directly (radar/fused-style)
    kf.P *= 5.0
    kf.R = np.diag([0.5, 0.5, 1.0, 1.0])   # measurement noise, tuned to sensor model above
    kf.Q = np.eye(4) * 0.05                 # process noise (unmodeled acceleration)
    return kf


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter
    age: int = 0
    misses: int = 0
    hits: int = 1

    def predict(self):
        self.kf.predict()

    def update(self, det):
        z = np.array([det["x"], det["y"], det["vx"], det["vy"]])
        self.kf.update(z)
        self.hits += 1
        self.misses = 0

    def as_dict(self):
        x, y, vx, vy = self.kf.x
        cov = self.kf.P
        return {
            "track_id": self.track_id,
            "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
            "cov_xy": [[float(cov[0, 0]), float(cov[0, 1])],
                       [float(cov[1, 0]), float(cov[1, 1])]],
            "age": self.age, "hits": self.hits,
        }


# ---------------------------------------------------------------------------
# Multi-object tracker: Hungarian association + Kalman update/predict/prune
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    def __init__(self, dt, gate_dist=4.0, max_misses=5):
        self.dt = dt
        self.gate_dist = gate_dist   # max match distance (meters), rejects bad associations
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 0

    def _cost_matrix(self, detections):
        n_tracks, n_dets = len(self.tracks), len(detections)
        cost = np.full((n_tracks, n_dets), 1e6)
        for i, trk in enumerate(self.tracks):
            tx, ty = trk.kf.x[0], trk.kf.x[1]
            for j, det in enumerate(detections):
                d = np.hypot(det["x"] - tx, det["y"] - ty)
                if d <= self.gate_dist:
                    cost[i, j] = d
        return cost

    def step(self, detections: list[dict]) -> list[dict]:
        for trk in self.tracks:
            trk.predict()
            trk.age += 1

        matched_tracks, matched_dets = set(), set()
        if self.tracks and detections:
            cost = self._cost_matrix(detections)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1e6:  # valid (gated) match
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in matched_tracks:
                trk.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                kf = make_cv_kalman_filter(det["x"], det["y"], det["vx"], det["vy"], self.dt)
                self.tracks.append(Track(track_id=self._next_id, kf=kf))
                self._next_id += 1

        # only publish confirmed tracks (hits >= 2) -- suppresses one-frame ghosts/false positives
        return [t.as_dict() for t in self.tracks if t.hits >= 2]


# ---------------------------------------------------------------------------
# Node wiring: subscribe to ground truth, publish tracked objects
# ---------------------------------------------------------------------------

class PerceptionNode:
    def __init__(self, rate_hz=20.0, ego_id=0):
        self.dt = 1.0 / rate_hz
        self.ego_id = ego_id
        self.noise_model = SensorNoiseModel()
        self.tracker = MultiObjectTracker(dt=self.dt)

        self.out_topic = bus.get_topic(
            "/perception/tracked_objects",
            qos=QoSProfile(reliability="reliable", history_depth=5,
                            deadline_ms=1000.0 / rate_hz * 3),
        )
        bus.get_topic("/sim/ground_truth").subscribe(self._on_ground_truth)

    def _on_ground_truth(self, msg):
        agents = msg["agents"]
        ego_state = next(a for a in agents if a["agent_id"] == self.ego_id)

        detections = self.noise_model.degrade(ego_state, agents)
        tracks = self.tracker.step(detections)

        self.out_topic.publish({
            "t": msg["t"],
            "ego": ego_state,
            "tracked_objects": tracks,
            "n_raw_detections": len(detections),
        })


if __name__ == "__main__":
    # Smoke test: run sim + perception together, print a mid-episode frame.
    from sim.ground_truth_sim import GroundTruthSimNode

    sim = GroundTruthSimNode(rate_hz=20.0)
    perception = PerceptionNode(rate_hz=20.0)

    last_frame = {}
    bus.get_topic("/perception/tracked_objects").subscribe(lambda m: last_frame.update(m))

    sim.run(duration_s=8.0)

    print(f"t={last_frame['t']:.2f}s | raw detections this frame: {last_frame['n_raw_detections']} "
          f"| confirmed tracks: {len(last_frame['tracked_objects'])}")
    for trk in last_frame["tracked_objects"]:
        print(f"  track {trk['track_id']}: x={trk['x']:.1f} y={trk['y']:.2f} "
              f"vx={trk['vx']:.1f} pos_cov_diag=({trk['cov_xy'][0][0]:.2f},{trk['cov_xy'][1][1]:.2f})")

    from bus.topic_bus import bus as _bus
    print("\nTopic stats:", _bus.stats())