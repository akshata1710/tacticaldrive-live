"""
topic_bus.py
------------
A minimal in-process pub/sub bus that mirrors ROS2's topic + QoS model closely
enough to prototype node architecture without a full ROS2 install.

Why this exists:
    ROS2/DDS gives you: named topics, multiple publishers/subscribers per topic,
    and QoS policies (reliability, deadline, history depth). This module gives
    you the SAME node-boundary discipline -- nodes only talk to each other via
    named topics, never via direct function calls or shared mutable state --
    so the architecture ports 1:1 to real rclpy later (see docs/ros2_port.md).

What this is NOT:
    A distributed system. No serialization, no network transport, no multi-process
    guarantees. It's a same-process stand-in so you can iterate on node logic fast.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class QoSProfile:
    """Mirrors the handful of ROS2 QoS settings that actually matter for a
    behavior-planning pipeline. See docs/ros2_port.md for the rclpy equivalents.
    """
    reliability: str = "reliable"   # "reliable" | "best_effort"
    history_depth: int = 10         # how many past messages a late subscriber can replay
    deadline_ms: float | None = None  # if set, missed deadlines are recorded (see Topic.deadline_misses)


@dataclass
class _Subscriber:
    callback: Callable[[Any], None]
    qos: QoSProfile


class Topic:
    """A single named topic. Publishers push messages; subscribers get called
    synchronously in publish order (good enough for single-process prototyping;
    real ROS2 gives you real scheduling/threading guarantees).
    """

    def __init__(self, name: str, qos: QoSProfile | None = None):
        self.name = name
        self.qos = qos or QoSProfile()
        self._subscribers: list[_Subscriber] = []
        self._history: list[Any] = []
        self._last_publish_time: float | None = None
        self.deadline_misses = 0
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Any], None], qos: QoSProfile | None = None):
        with self._lock:
            self._subscribers.append(_Subscriber(callback, qos or self.qos))
            # "durability"-lite: replay recent history to a fresh subscriber
            for msg in self._history[-self.qos.history_depth:]:
                callback(msg)

    def publish(self, message: Any):
        now = time.monotonic()
        with self._lock:
            if self.qos.deadline_ms is not None and self._last_publish_time is not None:
                gap_ms = (now - self._last_publish_time) * 1000.0
                if gap_ms > self.qos.deadline_ms:
                    self.deadline_misses += 1
            self._last_publish_time = now

            self._history.append(message)
            if len(self._history) > self.qos.history_depth:
                self._history.pop(0)

            subscribers = list(self._subscribers)

        for sub in subscribers:
            sub.callback(message)


class TopicBus:
    """Registry of topics, analogous to a ROS2 node graph / rclpy context."""

    def __init__(self):
        self._topics: dict[str, Topic] = {}

    def get_topic(self, name: str, qos: QoSProfile | None = None) -> Topic:
        if name not in self._topics:
            self._topics[name] = Topic(name, qos)
        return self._topics[name]

    def stats(self) -> dict[str, dict]:
        return {
            name: {
                "deadline_misses": t.deadline_misses,
                "n_messages": len(t._history),
                "n_subscribers": len(t._subscribers),
            }
            for name, t in self._topics.items()
        }


# Module-level singleton, mirroring how a ROS2 process shares one rclpy context.
bus = TopicBus()
