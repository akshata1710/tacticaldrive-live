"""
eval_tracking.py
-----------------
Runs a full episode through sim -> perception, then:

  1. Scores tracking quality against ground truth (position RMSE, false
     positive/negative rate, ID switches) -- this is what you'd report as
     "open-loop perception evaluation."
  2. Flags CRITICAL SCENARIOS using TTC (time-to-collision) and lateral-gap
     thresholds computed from ego + tracked (not ground-truth) state -- this
     is the "extract critical scenarios from fleet/sim data" deliverable,
     and it deliberately uses perception output (noisy) rather than ground
     truth, because that's what a real scenario-mining pipeline has to do.

Run:  PYTHONPATH=. python3 scripts/eval_tracking.py
"""

import numpy as np
from collections import defaultdict

from sim.ground_truth_sim import GroundTruthSimNode
from perception.perception_node import PerceptionNode
from bus.topic_bus import bus


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def run_episode(duration_s=8.0, rate_hz=20.0, seed=0):
    sim = GroundTruthSimNode(rate_hz=rate_hz, seed=seed)
    perception = PerceptionNode(rate_hz=rate_hz)

    ground_truth_log = []
    tracked_log = []

    bus.get_topic("/sim/ground_truth").subscribe(lambda m: ground_truth_log.append(m))
    bus.get_topic("/perception/tracked_objects").subscribe(lambda m: tracked_log.append(m))

    sim.run(duration_s=duration_s)
    return ground_truth_log, tracked_log


# ---------------------------------------------------------------------------
# Tracking metrics (simplified MOTA-style: matches nearest GT within gate)
# ---------------------------------------------------------------------------

def score_tracking(ground_truth_log, tracked_log, gate_dist=3.0):
    n_gt_total = 0
    n_matched = 0
    n_false_positive = 0
    n_false_negative = 0
    pos_errors = []

    track_to_gt_history = defaultdict(list)  # track_id -> list of matched gt agent_ids over time (for ID switch count)

    for gt_frame, trk_frame in zip(ground_truth_log, tracked_log):
        ego_id = trk_frame["ego"]["agent_id"]
        gt_agents = [a for a in gt_frame["agents"] if a["agent_id"] != ego_id]
        n_gt_total += len(gt_agents)

        gt_positions = {a["agent_id"]: (a["x"], a["y"]) for a in gt_agents}
        matched_gt_ids = set()

        for trk in trk_frame["tracked_objects"]:
            best_id, best_dist = None, gate_dist
            for gid, (gx, gy) in gt_positions.items():
                if gid in matched_gt_ids:
                    continue
                d = np.hypot(trk["x"] - gx, trk["y"] - gy)
                if d < best_dist:
                    best_id, best_dist = gid, d

            if best_id is not None:
                n_matched += 1
                matched_gt_ids.add(best_id)
                pos_errors.append(best_dist)
                track_to_gt_history[trk["track_id"]].append(best_id)
            else:
                n_false_positive += 1

        n_false_negative += len(gt_agents) - len(matched_gt_ids)

    # ID switches: a track_id that was matched to more than one distinct gt agent_id over the episode
    id_switches = sum(1 for hist in track_to_gt_history.values() if len(set(hist)) > 1)

    return {
        "n_gt_total": n_gt_total,
        "n_matched": n_matched,
        "n_false_positive": n_false_positive,
        "n_false_negative": n_false_negative,
        "position_rmse_m": float(np.sqrt(np.mean(np.square(pos_errors)))) if pos_errors else None,
        "id_switches": id_switches,
        "recall": n_matched / n_gt_total if n_gt_total else None,
    }


# ---------------------------------------------------------------------------
# Critical scenario extraction (TTC-based, uses PERCEPTION output, not GT)
# ---------------------------------------------------------------------------

def extract_critical_scenarios(tracked_log, ttc_threshold_s=2.0):
    """Flags frames where any tracked object is on a closing trajectory with
    ego within `ttc_threshold_s`. Uses a simple longitudinal TTC approximation
    (same-lane closing rate); a production version would do full 2D TTC.
    """
    critical_frames = []

    for frame in tracked_log:
        ego = frame["ego"]
        for trk in frame["tracked_objects"]:
            same_lane = abs(trk["y"] - ego["y"]) < 1.8   # rough lane-width gate
            closing_rate = ego["vx"] - trk["vx"]         # ego catching up to a slower lead, or vice versa
            gap = trk["x"] - ego["x"]

            if same_lane and gap > 0 and closing_rate > 0:
                ttc = gap / closing_rate
                if ttc < ttc_threshold_s:
                    critical_frames.append({
                        "t": frame["t"], "track_id": trk["track_id"],
                        "ttc_s": round(ttc, 2), "gap_m": round(gap, 1),
                    })

    return critical_frames


if __name__ == "__main__":
    gt_log, trk_log = run_episode(duration_s=8.0, rate_hz=20.0, seed=0)

    metrics = score_tracking(gt_log, trk_log)
    print("=== Tracking metrics (open-loop, vs. ground truth) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    critical = extract_critical_scenarios(trk_log, ttc_threshold_s=2.0)
    print(f"\n=== Critical scenarios flagged (TTC < 2.0s): {len(critical)} frames ===")
    for c in critical[:10]:
        print(f"  t={c['t']:.2f}s  track={c['track_id']}  TTC={c['ttc_s']}s  gap={c['gap_m']}m")
    if len(critical) > 10:
        print(f"  ... and {len(critical) - 10} more")
