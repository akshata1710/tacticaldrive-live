"""
ground_truth_sim.py
--------------------
A minimal multi-agent kinematic-bicycle simulator that publishes ground-truth
vehicle states onto /sim/ground_truth at a fixed rate (mirrors a sensor/world
publisher node in a real AD stack).

Scenario: a highway merge -- ego vehicle plus N surrounding agents, one of
which cuts in front of ego partway through the episode. This gives the
perception + tracking layer something non-trivial to handle (an agent that
changes lane / relative position mid-episode).

Ground truth is intentionally "too good" -- perception_node.py is what
degrades this into something realistic (noise, dropout, false negatives),
which is the actual point of this project.
"""

import time
import math
import numpy as np
from dataclasses import dataclass, asdict

from bus.topic_bus import bus, QoSProfile


@dataclass
class AgentState:
    agent_id: int
    x: float          # meters, along-road
    y: float           # meters, lateral (lane offset)
    vx: float          # m/s
    vy: float
    heading: float      # radians
    length: float = 4.5
    width: float = 1.9
    cls: str = "car"

    def as_dict(self):
        return asdict(self)


# Cut-in-and-brake scenario for agent 2: start x (m), brake start (s), decel (m/s^2), floor speed (m/s).
# Tuned so the default episode produces one clear near-miss (min TTC ~1 s) for scenario mining.
BRAKE_X0, BRAKE_T, BRAKE_DECEL, BRAKE_MIN = 10.0, 4.5, 4.0, 19.0


class KinematicAgent:
    """Constant-velocity-with-optional-lateral-maneuver agent."""

    def __init__(self, agent_id, x, y, speed, lateral_maneuver=None, braking=None):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.vx = speed
        self.vy = 0.0
        self.heading = 0.0
        # lateral_maneuver: (t_start, t_end, target_y) -- a lane change window
        self.lateral_maneuver = lateral_maneuver
        # braking: (t_start, decel_mps2, min_speed) -- longitudinal braking event
        self.braking = braking

    def step(self, t, dt):
        if self.lateral_maneuver:
            t0, t1, target_y = self.lateral_maneuver
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                # smoothstep for a plausible lane-change lateral profile
                s = 3 * frac**2 - 2 * frac**3
                self.vy = (target_y - self.y) / max(t1 - t, 1e-3)
                self.y += self.vy * dt
                self.heading = math.atan2(self.vy, max(self.vx, 1e-3))
            elif t > t1:
                self.vy = 0.0
                self.heading = 0.0
        if self.braking:
            t_brake, decel, min_speed = self.braking
            if t >= t_brake and self.vx > min_speed:
                self.vx = max(min_speed, self.vx - decel * dt)
        self.x += self.vx * dt

    def state(self, t) -> AgentState:
        return AgentState(
            agent_id=self.agent_id, x=self.x, y=self.y,
            vx=self.vx, vy=self.vy, heading=self.heading,
        )


class GroundTruthSimNode:
    """Publishes list[AgentState] to /sim/ground_truth at `rate_hz`."""

    def __init__(self, rate_hz: float = 20.0, seed: int = 0):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.t = 0.0
        rng = np.random.default_rng(seed)

        # ego + 4 agents. Agent 2 overtakes, cuts in front of ego between t=2s and t=4s,
        # then brakes hard: the classic cut-in-and-brake case that scenario mining should flag.
        self.agents = [
            KinematicAgent(agent_id=0, x=0.0,  y=0.0, speed=25.0),                     # ego
            KinematicAgent(agent_id=1, x=45.0, y=0.0, speed=24.0),                     # lead, same lane
            KinematicAgent(agent_id=2, x=BRAKE_X0, y=3.5, speed=27.0,
                            lateral_maneuver=(2.0, 4.0, 0.0),
                            braking=(BRAKE_T, BRAKE_DECEL, BRAKE_MIN)),                 # cut-in + brake
            KinematicAgent(agent_id=3, x=-10.0, y=-3.5, speed=23.0),                    # trailing, other lane
            KinematicAgent(agent_id=4, x=70.0, y=0.0, speed=26.0),                      # far lead
        ]

        self.topic = bus.get_topic(
            "/sim/ground_truth",
            qos=QoSProfile(reliability="reliable", history_depth=5, deadline_ms=1000.0 / rate_hz * 3),
        )

    def step(self):
        for a in self.agents:
            a.step(self.t, self.dt)
        states = [a.state(self.t).as_dict() for a in self.agents]
        self.topic.publish({"t": self.t, "agents": states})
        self.t += self.dt

    def run(self, duration_s: float, realtime: bool = False):
        n_steps = int(duration_s * self.rate_hz)
        for _ in range(n_steps):
            self.step()
            if realtime:
                time.sleep(self.dt)


if __name__ == "__main__":
    node = GroundTruthSimNode(rate_hz=20.0)
    node.run(duration_s=8.0)
    print(f"Published {len(node.topic._history)} buffered ground-truth frames "
          f"(history_depth caps buffer; full run was {int(8.0*20)} frames).")
