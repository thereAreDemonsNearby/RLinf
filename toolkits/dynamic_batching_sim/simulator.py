"""GPU-robot dynamic batching simulator.

Discrete-event simulation of N GPUs serving M robots with batched inference,
based on real pi05 model timing data. Produces statistics and an HTML
visualization with Gantt charts.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Simulation configuration parameters."""

    num_gpus: int = 4
    num_robots: int = 8
    max_batch_size: int = 16  # must be even, <= 32
    idle_time_threshold: float = 0.010  # max wait time to collect observations (s)
    network_latency: float = 0.005  # fixed latency for upload and download (s)
    num_rounds: int = 10
    action_time_min: float = 0.035  # min time per robot action (s)
    action_time_max: float = 0.045  # max time per robot action (s)
    actions_per_block: int = 30
    csv_path: str = ""  # path to speed_pi05.csv; auto-detected if empty

    def __post_init__(self):
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
        if self.num_robots < 1:
            raise ValueError("num_robots must be >= 1")
        if self.max_batch_size < 1 or self.max_batch_size > 32:
            raise ValueError("max_batch_size must be in [1, 32]")
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if self.action_time_min > self.action_time_max:
            raise ValueError("action_time_min must be <= action_time_max")
        if self.idle_time_threshold < 0:
            raise ValueError("idle_time_threshold must be >= 0")
        if self.network_latency < 0:
            raise ValueError("network_latency must be >= 0")


# ---------------------------------------------------------------------------
# Inference timing model
# ---------------------------------------------------------------------------


class InferenceTimingModel:
    """Interpolation model for inference times at different batch sizes.

    Loads measured data from speed_pi05.csv and linearly interpolates
    for batch sizes not explicitly measured.
    """

    def __init__(self, csv_path: str):
        self._batch_sizes = np.array([], dtype=np.float64)
        self._vision_times = np.array([], dtype=np.float64)
        self._lm_times = np.array([], dtype=np.float64)
        self._denoise_times = np.array([], dtype=np.float64)
        self._total_times = np.array([], dtype=np.float64)
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        bs_list, vt_list, lt_list, dt_list, tt_list = [], [], [], [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bs = int(row["batch_size"])
                bs_list.append(bs)
                vt_list.append(float(row["vision_time"]))
                lt_list.append(float(row["lm_time"]))
                dt_list.append(float(row["denoise_time"]))
                tt_list.append(float(row["total_time"]))
        order = np.argsort(bs_list)
        self._batch_sizes = np.array(bs_list, dtype=np.float64)[order]
        self._vision_times = np.array(vt_list, dtype=np.float64)[order]
        self._lm_times = np.array(lt_list, dtype=np.float64)[order]
        self._denoise_times = np.array(dt_list, dtype=np.float64)[order]
        self._total_times = np.array(tt_list, dtype=np.float64)[order]

    def get_times(self, batch_size: int) -> Dict[str, float]:
        """Return interpolated timing for a given batch size.

        Args:
            batch_size: Number of observations in the batch (1–32).

        Returns:
            Dict with keys 'vision', 'lm', 'denoise', 'total' (all in seconds).
        """
        bs = float(max(1, min(32, batch_size)))
        xp = self._batch_sizes
        vision = float(np.interp(bs, xp, self._vision_times))
        lm = float(np.interp(bs, xp, self._lm_times))
        denoise = float(np.interp(bs, xp, self._denoise_times))
        total = float(np.interp(bs, xp, self._total_times))
        return {"vision": vision, "lm": lm, "denoise": denoise, "total": total}


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------


@dataclass(order=True)
class Event:
    """A simulation event, ordered by scheduled time."""

    time: float
    event_type: str = field(compare=False)
    gpu_id: int = field(default=-1, compare=False)
    robot_id: int = field(default=-1, compare=False)
    round_id: int = field(default=-1, compare=False)
    payload: dict = field(default_factory=dict, compare=False)


@dataclass
class GPUState:
    """Per-GPU state during simulation."""

    gpu_id: int
    state: str = "IDLE"  # "IDLE" | "BUSY"
    current_batch: List[Tuple[int, int]] = field(default_factory=list)
    async_collected: List[Tuple[int, int]] = field(default_factory=list)
    idle_start_time: Optional[float] = None  # when GPU became idle
    inference_start_time: float = 0.0
    total_busy_time: float = 0.0
    # For Gantt: record state transitions
    history: List[dict] = field(default_factory=list)


@dataclass
class RobotState:
    """Per-robot state during simulation."""

    robot_id: int
    state: str = "IDLE"  # "IDLE" | "WAITING" | "EXECUTING" | "DONE"
    current_round: int = 0
    actions_completed: int = 0
    assigned_gpu_id: int = -1
    wait_start_time: float = 0.0
    history: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------


class BatchingSimulator:
    """Discrete-event simulator for GPU-robot dynamic batching."""

    def __init__(self, config: SimConfig):
        self.config = config
        # Resolve CSV path
        if config.csv_path:
            csv_path = config.csv_path
        else:
            csv_path = str(
                Path(__file__).resolve().parent.parent.parent / "speed_pi05.csv"
            )
        self.timing = InferenceTimingModel(csv_path)

        # State
        self.gpus: List[GPUState] = [
            GPUState(gpu_id=i) for i in range(config.num_gpus)
        ]
        self.robots: List[RobotState] = [
            RobotState(robot_id=i) for i in range(config.num_robots)
        ]
        self.event_queue: List[Event] = []
        self.current_time: float = 0.0

        # Statistics
        self.robot_wait_times: List[float] = []
        self.inference_count: int = 0
        self.total_obs_processed: int = 0

        # Tie-break counter for round-robin GPU assignment
        self._gpu_assign_counter: int = 0

        # Detailed timeline for Gantt
        self.gpu_timeline: List[dict] = []  # {gpu_id, start, end, type, stage}
        self.robot_timeline: List[dict] = []  # {robot_id, start, end, state, round}

    # ------------------------------------------------------------------
    # Event queue helpers
    # ------------------------------------------------------------------

    def _schedule(self, event: Event):
        heapq.heappush(self.event_queue, event)

    def _schedule_obs_arrival(
        self, robot_id: int, round_id: int, gpu_id: int, delay: float
    ):
        self._schedule(
            Event(
                time=self.current_time + delay,
                event_type="obs_arrives_at_gpu",
                robot_id=robot_id,
                round_id=round_id,
                gpu_id=gpu_id,
            )
        )

    # ------------------------------------------------------------------
    # GPU selection
    # ------------------------------------------------------------------

    def _select_least_busy_gpu(self) -> int:
        """Return the GPU ID with the fewest queued + processing observations.

        Ties are broken round-robin for even initial distribution.
        """
        def load(gpu_id: int) -> int:
            g = self.gpus[gpu_id]
            return len(g.current_batch) + len(g.async_collected)

        min_load = min(load(i) for i in range(self.config.num_gpus))
        candidates = [i for i in range(self.config.num_gpus) if load(i) == min_load]

        if len(candidates) == 1:
            return candidates[0]
        # Round-robin tie-break
        idx = self._gpu_assign_counter % len(candidates)
        self._gpu_assign_counter += 1
        return candidates[idx]

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_last_entry(history: List[dict], close_time: float):
        """Set end time on the most recent history entry if it is still open."""
        if history and history[-1]["end"] is None:
            history[-1]["end"] = close_time

    # ------------------------------------------------------------------
    # GPU inference
    # ------------------------------------------------------------------

    def _start_gpu_inference(self, gpu_id: int):
        """Start inference on a GPU with its current batch.

        Caps the batch at max_batch_size; leftover observations stay in
        current_batch for the next round.
        """
        gpu = self.gpus[gpu_id]
        all_obs = list(gpu.current_batch)
        if len(all_obs) == 0:
            return

        # Cap batch size; remainder stays queued for next inference
        max_bs = self.config.max_batch_size
        batch = all_obs[:max_bs]
        remaining = all_obs[max_bs:]
        batch_size = len(batch)

        times = self.timing.get_times(batch_size)
        total_time = times["total"]
        vision_time = times["vision"]
        lm_time = times["lm"]

        # Record GPU state transition
        prev_state = gpu.state
        t0 = self.current_time

        gpu.state = "BUSY"
        gpu.inference_start_time = t0
        gpu.async_collected = []
        gpu.current_batch = remaining

        # Close previous IDLE period
        self._close_last_entry(gpu.history, t0)

        # Record BUSY period for Gantt
        gpu.history.append(
            {"start": t0, "end": t0 + total_time, "state": "BUSY",
             "stage": "busy", "batch_size": batch_size}
        )

        self.inference_count += 1
        self.total_obs_processed += batch_size

        # Schedule inference completion
        self._schedule(
            Event(
                time=t0 + total_time,
                event_type="gpu_inference_done",
                gpu_id=gpu_id,
                payload={"batch": batch, "batch_size": batch_size},
            )
        )

    def _check_and_start_inference(self, gpu_id: int):
        """Start inference on a GPU if conditions are met."""
        gpu = self.gpus[gpu_id]
        if gpu.state != "IDLE":
            return
        batch_size = len(gpu.current_batch)
        if batch_size == 0:
            return

        idle_time = self.current_time - (gpu.idle_start_time or self.current_time)

        if batch_size >= self.config.max_batch_size:
            self._start_gpu_inference(gpu_id)
        elif idle_time >= self.config.idle_time_threshold:
            # Soft even constraint: allow odd batch when timeout expires
            self._start_gpu_inference(gpu_id)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_obs_arrives_at_gpu(self, event: Event):
        robot_id = event.robot_id
        round_id = event.round_id
        gpu_id = event.gpu_id
        gpu = self.gpus[gpu_id]
        robot = self.robots[robot_id]

        # The observation (robot_id, round_id) is already "in flight" and
        # arrives at the GPU now.  Add it to the appropriate collection.
        if gpu.state == "BUSY":
            gpu.async_collected.append((robot_id, round_id))
        else:
            # IDLE
            was_empty = len(gpu.current_batch) == 0
            gpu.current_batch.append((robot_id, round_id))
            if was_empty:
                # First observation in this idle period — start idle timer.
                # (IDLE history entry was already added when GPU entered IDLE.)
                gpu.idle_start_time = self.current_time
                self._schedule(
                    Event(
                        time=self.current_time + self.config.idle_time_threshold,
                        event_type="check_idle_timeout",
                        gpu_id=gpu_id,
                    )
                )
            self._check_and_start_inference(gpu_id)

    def _handle_gpu_inference_done(self, event: Event):
        gpu_id = event.gpu_id
        gpu = self.gpus[gpu_id]
        batch = event.payload["batch"]
        t_finish = self.current_time

        # Accumulate busy time
        gpu.total_busy_time += t_finish - gpu.inference_start_time

        # Close the last BUSY history entry (denoise stage ends at current time)
        # The history entries already have correct end times from _start_gpu_inference

        # Send action blocks back to each robot in the batch
        for (robot_id, round_id) in batch:
            self._schedule(
                Event(
                    time=t_finish + self.config.network_latency,
                    event_type="action_arrives_at_robot",
                    robot_id=robot_id,
                    round_id=round_id,
                    payload={"action_count": self.config.actions_per_block},
                )
            )

        # Append async-collected observations to current_batch (which may
        # already hold leftovers from a previous capped batch).
        gpu.current_batch.extend(gpu.async_collected)
        gpu.async_collected = []

        if len(gpu.current_batch) > 0:
            # Check if we should start inference immediately
            if len(gpu.current_batch) >= self.config.max_batch_size:
                # Start immediately — stay BUSY
                self._start_gpu_inference(gpu_id)
            else:
                # Go IDLE but with existing observations
                gpu.state = "IDLE"
                gpu.idle_start_time = t_finish
                self._close_last_entry(gpu.history, t_finish)
                gpu.history.append(
                    {"start": t_finish, "end": None,
                     "state": "IDLE", "stage": "idle", "batch_size": 0}
                )
                self._schedule(
                    Event(
                        time=t_finish + self.config.idle_time_threshold,
                        event_type="check_idle_timeout",
                        gpu_id=gpu_id,
                    )
                )
        else:
            # Go truly IDLE with empty queue
            gpu.state = "IDLE"
            gpu.idle_start_time = None
            self._close_last_entry(gpu.history, t_finish)
            gpu.history.append(
                {"start": t_finish, "end": None,
                 "state": "IDLE", "stage": "idle", "batch_size": 0}
            )

    def _handle_action_arrives_at_robot(self, event: Event):
        robot_id = event.robot_id
        round_id = event.round_id
        robot = self.robots[robot_id]

        # Record wait time
        wait_time = self.current_time - robot.wait_start_time
        self.robot_wait_times.append(wait_time)

        # Begin executing the action block
        robot.state = "EXECUTING"
        robot.actions_completed = 0

        # Close previous WAITING period, then start EXECUTING
        self._close_last_entry(robot.history, self.current_time)
        robot.history.append(
            {"start": self.current_time, "end": None,
             "state": "EXECUTING", "round": round_id}
        )

        # Schedule first action
        action_dur = random.uniform(
            self.config.action_time_min, self.config.action_time_max
        )
        self._schedule(
            Event(
                time=self.current_time + action_dur,
                event_type="robot_action_done",
                robot_id=robot_id,
                round_id=round_id,
                payload={"action_index": 0},
            )
        )

    def _handle_robot_action_done(self, event: Event):
        robot_id = event.robot_id
        round_id = event.round_id
        action_index = event.payload["action_index"]
        robot = self.robots[robot_id]

        robot.actions_completed += 1

        if robot.actions_completed < self.config.actions_per_block:
            # Execute next action
            action_dur = random.uniform(
                self.config.action_time_min, self.config.action_time_max
            )
            self._schedule(
                Event(
                    time=self.current_time + action_dur,
                    event_type="robot_action_done",
                    robot_id=robot_id,
                    round_id=round_id,
                    payload={"action_index": action_index + 1},
                )
            )
        else:
            # All actions done for this round
            # Close EXECUTING history entry
            self._close_last_entry(robot.history, self.current_time)

            if round_id + 1 < self.config.num_rounds:
                # Start next round
                next_round = round_id + 1
                gpu_id = self._select_least_busy_gpu()
                robot.assigned_gpu_id = gpu_id
                robot.current_round = next_round
                robot.state = "WAITING"
                robot.wait_start_time = self.current_time

                # Close previous state, then record WAITING
                self._close_last_entry(robot.history, self.current_time)
                robot.history.append(
                    {"start": self.current_time, "end": None,
                     "state": "WAITING", "round": next_round}
                )

                # Observation travels over network to GPU.
                # The GPU will add it when obs_arrives_at_gpu fires.
                self._schedule_obs_arrival(
                    robot_id, next_round, gpu_id, self.config.network_latency
                )
            else:
                # All rounds complete for this robot
                robot.state = "DONE"

    def _handle_check_idle_timeout(self, event: Event):
        gpu_id = event.gpu_id
        gpu = self.gpus[gpu_id]

        if gpu.state != "IDLE":
            return  # No-op: GPU is already busy
        if len(gpu.current_batch) == 0:
            return  # No-op: nothing to process

        # Timeout expired — start inference with whatever we have
        # (soft even constraint: odd batch sizes allowed here)
        self._start_gpu_inference(gpu_id)

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the simulation and return statistics."""
        self._initialize()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == "obs_arrives_at_gpu":
                self._handle_obs_arrives_at_gpu(event)
            elif event.event_type == "gpu_inference_done":
                self._handle_gpu_inference_done(event)
            elif event.event_type == "action_arrives_at_robot":
                self._handle_action_arrives_at_robot(event)
            elif event.event_type == "robot_action_done":
                self._handle_robot_action_done(event)
            elif event.event_type == "check_idle_timeout":
                self._handle_check_idle_timeout(event)

        # End-of-simulation flush: handle any IDLE GPUs with pending observations
        self._flush_remaining()

        # Finalize Gantt timelines
        self._finalize_timelines()

        return self._compute_statistics()

    def _initialize(self):
        """Set up initial events: all robots send their first observation."""
        for robot_id in range(self.config.num_robots):
            gpu_id = self._select_least_busy_gpu()
            robot = self.robots[robot_id]
            robot.assigned_gpu_id = gpu_id
            robot.state = "WAITING"
            robot.current_round = 0
            robot.wait_start_time = 0.0

            robot.history.append(
                {"start": 0.0, "end": None,
                 "state": "WAITING", "round": 0}
            )

            # Observation travels over network; GPU handler adds it on arrival
            self._schedule_obs_arrival(
                robot_id, 0, gpu_id, self.config.network_latency
            )

        # GPUs are idle initially (idle timer starts when first obs arrives)
        for gpu in self.gpus:
            gpu.history.append(
                {"start": 0.0, "end": None,
                 "state": "IDLE", "stage": "idle", "batch_size": 0}
            )

    def _flush_remaining(self):
        """Flush any GPUs that still have pending observations."""
        for gpu in self.gpus:
            if gpu.state == "IDLE" and len(gpu.current_batch) > 0:
                self._start_gpu_inference(gpu.gpu_id)
            # Move async-collected to current if BUSY and flush after done
            elif gpu.state == "BUSY" and len(gpu.async_collected) > 0:
                # We need to wait for current inference to finish, then process
                # the async-collected observations. Schedule a flush event.
                pass  # Will be handled by gpu_inference_done which moves async_collected

        # Process remaining events from flush
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = max(self.current_time, event.time)

            if event.event_type == "obs_arrives_at_gpu":
                self._handle_obs_arrives_at_gpu(event)
            elif event.event_type == "gpu_inference_done":
                self._handle_gpu_inference_done(event)
            elif event.event_type == "action_arrives_at_robot":
                self._handle_action_arrives_at_robot(event)
            elif event.event_type == "robot_action_done":
                self._handle_robot_action_done(event)
            elif event.event_type == "check_idle_timeout":
                self._handle_check_idle_timeout(event)

    def _finalize_timelines(self):
        """Build consolidated Gantt timelines from GPU and robot histories."""
        for gpu in self.gpus:
            for entry in gpu.history:
                self.gpu_timeline.append(
                    {
                        "gpu_id": gpu.gpu_id,
                        "start": entry["start"],
                        "end": entry["end"] if entry["end"] is not None else self.current_time,
                        "state": entry["state"],
                        "stage": entry.get("stage", ""),
                        "batch_size": entry.get("batch_size", 0),
                    }
                )
        for robot in self.robots:
            for entry in robot.history:
                self.robot_timeline.append(
                    {
                        "robot_id": robot.robot_id,
                        "start": entry["start"],
                        "end": entry["end"] if entry["end"] is not None else self.current_time,
                        "state": entry["state"],
                        "round": entry.get("round", 0),
                    }
                )

    def _compute_statistics(self) -> dict:
        """Compute summary statistics."""
        total_time = self.current_time

        # GPU utilization per GPU and overall
        gpu_utils = []
        for gpu in self.gpus:
            util = gpu.total_busy_time / total_time * 100.0 if total_time > 0 else 0.0
            gpu_utils.append(
                {"gpu_id": gpu.gpu_id, "busy_time": gpu.total_busy_time,
                 "utilization_pct": util}
            )
        overall_util = (
            sum(g.total_busy_time for g in self.gpus)
            / (total_time * self.config.num_gpus) * 100.0
            if total_time > 0
            else 0.0
        )

        # Robot wait time percentiles
        waits = np.array(self.robot_wait_times) if self.robot_wait_times else np.array([0.0])
        p50 = float(np.percentile(waits, 50))
        p90 = float(np.percentile(waits, 90))
        p99 = float(np.percentile(waits, 99))
        mean_wait = float(np.mean(waits))
        max_wait = float(np.max(waits))
        min_wait = float(np.min(waits))

        return {
            "total_time": total_time,
            "num_inferences": self.inference_count,
            "total_obs_processed": self.total_obs_processed,
            "gpu_utils": gpu_utils,
            "overall_utilization_pct": overall_util,
            "wait_p50": p50,
            "wait_p90": p90,
            "wait_p99": p99,
            "wait_mean": mean_wait,
            "wait_max": max_wait,
            "wait_min": min_wait,
            "num_waits_recorded": len(self.robot_wait_times),
        }

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def generate_html(self, stats: dict, output_path: str):
        """Generate a self-contained HTML report with ECharts visualizations."""
        # Serialize timeline data for embedding
        gpu_timeline_json = json.dumps(self.gpu_timeline)
        robot_timeline_json = json.dumps(self.robot_timeline)
        stats_json = json.dumps(stats, indent=2)
        config_json = json.dumps(
            {
                "num_gpus": self.config.num_gpus,
                "num_robots": self.config.num_robots,
                "max_batch_size": self.config.max_batch_size,
                "idle_time_threshold": self.config.idle_time_threshold,
                "network_latency": self.config.network_latency,
                "num_rounds": self.config.num_rounds,
                "action_time_min": self.config.action_time_min,
                "action_time_max": self.config.action_time_max,
                "actions_per_block": self.config.actions_per_block,
            },
            indent=2,
        )

        html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dynamic Batching Simulation Report</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
        <style>
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                 background: #f5f7fa; color: #333; padding: 20px; }}
          h1 {{ text-align: center; margin-bottom: 10px; color: #1a1a2e; }}
          .config-box {{ max-width: 900px; margin: 0 auto 20px; background: #fff; border-radius: 8px;
                         padding: 16px 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
          .config-box h3 {{ margin-bottom: 8px; }}
          .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 4px 16px; }}
          .config-grid .item {{ font-size: 13px; }}
          .config-grid .item span:first-child {{ color: #666; }}
          .config-grid .item span:last-child {{ font-weight: 600; }}
          .stats-box {{ max-width: 900px; margin: 0 auto 20px; background: #fff; border-radius: 8px;
                        padding: 16px 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
          .stats-box h3 {{ margin-bottom: 8px; }}
          .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px 16px; }}
          .chart-container {{ max-width: 1200px; margin: 0 auto 20px; background: #fff;
                              border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
          .chart {{ width: 100%; height: 500px; }}
          .chart-wide {{ width: 100%; height: 600px; }}
          .chart-half {{ width: 100%; height: 400px; }}
          .row {{ display: flex; gap: 20px; max-width: 1200px; margin: 0 auto; }}
          .row .chart-container {{ flex: 1; }}
          @media (max-width: 800px) {{ .row {{ flex-direction: column; }} }}
        </style>
        </head>
        <body>
        <h1>Dynamic Batching Simulation Report</h1>

        <div class="config-box">
          <h3>Configuration</h3>
          <div class="config-grid" id="configGrid"></div>
        </div>

        <div class="stats-box">
          <h3>Summary Statistics</h3>
          <div class="stats-grid" id="statsGrid"></div>
        </div>

        <div class="chart-container">
          <h3>GPU Gantt Chart</h3>
          <div class="chart-wide" id="gpuGantt"></div>
        </div>

        <div class="chart-container">
          <h3>Robot Gantt Chart</h3>
          <div class="chart-wide" id="robotGantt"></div>
        </div>

        <div class="row">
          <div class="chart-container">
            <h3>GPU Utilization</h3>
            <div class="chart-half" id="utilPie"></div>
          </div>
          <div class="chart-container">
            <h3>Robot Wait Time (seconds)</h3>
            <div class="chart-half" id="waitBar"></div>
          </div>
        </div>

        <script>
        const gpuTimeline = {gpu_timeline_json};
        const robotTimeline = {robot_timeline_json};
        const stats = {stats_json};
        const config = {config_json};

        // --- Config display ---
        (function() {{
          const grid = document.getElementById('configGrid');
          const items = [
            ['GPUs', config.num_gpus],
            ['Robots', config.num_robots],
            ['Max Batch Size', config.max_batch_size],
            ['Idle Threshold', config.idle_time_threshold.toFixed(3) + ' s'],
            ['Network Latency', config.network_latency.toFixed(3) + ' s'],
            ['Rounds', config.num_rounds],
            ['Action Time', config.action_time_min.toFixed(3) + '–' + config.action_time_max.toFixed(3) + ' s'],
            ['Actions/Block', config.actions_per_block],
          ];
          items.forEach(([label, val]) => {{
            grid.innerHTML += '<div class="item"><span>' + label + ': </span><span>' + val + '</span></div>';
          }});
        }})();

        // --- Stats display ---
        (function() {{
          const grid = document.getElementById('statsGrid');
          const items = [
            ['Total Sim Time', stats.total_time.toFixed(3) + ' s'],
            ['Inferences Run', stats.num_inferences],
            ['Obs Processed', stats.total_obs_processed],
            ['Overall GPU Util', stats.overall_utilization_pct.toFixed(1) + '%'],
            ['Wait P50', stats.wait_p50.toFixed(4) + ' s'],
            ['Wait P90', stats.wait_p90.toFixed(4) + ' s'],
            ['Wait P99', stats.wait_p99.toFixed(4) + ' s'],
            ['Wait Mean', stats.wait_mean.toFixed(4) + ' s'],
            ['Wait Min', stats.wait_min.toFixed(4) + ' s'],
            ['Wait Max', stats.wait_max.toFixed(4) + ' s'],
          ];
          items.forEach(([label, val]) => {{
            grid.innerHTML += '<div class="item"><span>' + label + ': </span><span>' + val + '</span></div>';
          }});
        }})();

        // --- GPU Gantt chart ---
        (function() {{
          const chart = echarts.init(document.getElementById('gpuGantt'));
          const gpus = [...new Set(gpuTimeline.map(d => d.gpu_id))].sort((a,b) => a - b);
          const gpuStateColors = {{ busy: '#5470c6', idle: '#e0e0e0' }};
          const gpuStateNames = {{ busy: 'Busy', idle: 'Idle' }};

          function renderGpuBar(params, api) {{
            const gpuIdx = api.value(0);
            const start = api.coord([api.value(1), gpuIdx]);
            const end = api.coord([api.value(2), gpuIdx]);
            const height = api.size([0, 1])[1] * 0.6;
            const y = start[1] - height / 2;
            return {{
              type: 'rect',
              shape: {{ x: start[0], y: y, width: Math.max(end[0] - start[0], 1), height: height }},
              style: api.style(),
              styleEmphasis: api.styleEmphasis(),
            }};
          }}

          // Render Idle first, then Busy on top (otherwise wide Idle bars cover Busy)
          const gpuSeries = ['idle', 'busy'].map(function(st) {{
            const filtered = gpuTimeline
              .filter(d => (d.stage || d.state.toLowerCase()) === st)
              .map(d => ({{
                value: [d.gpu_id, d.start, d.end, st, d.batch_size],
                itemStyle: {{ color: gpuStateColors[st] || '#999' }}
              }}));
            return {{
              name: gpuStateNames[st],
              type: 'custom',
              renderItem: renderGpuBar,
              encode: {{ x: [1, 2], y: 0 }},
              data: filtered,
            }};
          }});

          const gpuOption = {{
            tooltip: {{
              trigger: 'item',
              formatter: function(p) {{
                const d = p.data.value;
                const dur = (d[2] - d[1]).toFixed(4);
                const bs = d[4];
                var html = '<b>GPU ' + d[0] + '</b><br/>'
                  + 'State: ' + (gpuStateNames[d[3]] || d[3]) + '<br/>'
                  + 'Start: ' + d[1].toFixed(4) + ' s<br/>'
                  + 'End: ' + d[2].toFixed(4) + ' s<br/>'
                  + 'Duration: ' + dur + ' s';
                if (d[3] === 'busy' && bs > 0) {{
                  html += '<br/>Batch Size: <b>' + bs + '</b>';
                }}
                return html;
              }}
            }},
            legend: {{
              data: ['Busy', 'Idle'],
              orient: 'horizontal',
              left: 'center',
              top: 0,
              textStyle: {{ fontSize: 11 }},
            }},
            grid: {{ left: 60, right: 30, top: 35, bottom: 30 }},
            xAxis: {{ type: 'value', name: 'Time (s)', nameLocation: 'center', nameGap: 25,
                      axisLabel: {{ formatter: v => v.toFixed(1) }} }},
            yAxis: {{ type: 'category', name: 'GPU', data: gpus.map(g => 'GPU ' + g),
                      inverse: true }},
            series: gpuSeries,
          }};
          chart.setOption(gpuOption);
          window.addEventListener('resize', () => chart.resize());
        }})();

        // --- Robot Gantt chart ---
        (function() {{
          const chart = echarts.init(document.getElementById('robotGantt'));
          const robots = [...new Set(robotTimeline.map(d => d.robot_id))].sort((a,b) => a - b);
          const robotStateDefs = [
            {{ key: 'WAITING',   name: 'Waiting for GPU',    color: '#fc8452' }},
            {{ key: 'EXECUTING', name: 'Executing Actions',  color: '#73c0de' }},
            {{ key: 'DONE',      name: 'Done',               color: '#9a60b4' }},
            {{ key: 'IDLE',      name: 'Idle',               color: '#e0e0e0' }},
          ];
          const robotColorMap = {{}};
          const robotNameMap = {{}};
          robotStateDefs.forEach(s => {{ robotColorMap[s.key] = s.color; robotNameMap[s.key] = s.name; }});

          // Shared renderItem for all Robot series
          function renderRobotBar(params, api) {{
            const robotIdx = api.value(0);
            const start = api.coord([api.value(1), robotIdx]);
            const end = api.coord([api.value(2), robotIdx]);
            const height = api.size([0, 1])[1] * 0.6;
            const y = start[1] - height / 2;
            return {{
              type: 'rect',
              shape: {{ x: start[0], y: y, width: Math.max(end[0] - start[0], 1), height: height }},
              style: api.style(),
              styleEmphasis: api.styleEmphasis(),
            }};
          }}

          const robotSeries = robotStateDefs.map(function(sd) {{
            const filtered = robotTimeline
              .filter(d => d.state === sd.key)
              .map(d => ({{
                value: [d.robot_id, d.start, d.end, d.state, d.round],
                itemStyle: {{ color: sd.color }}
              }}));
            return {{
              name: sd.name,
              type: 'custom',
              renderItem: renderRobotBar,
              encode: {{ x: [1, 2], y: 0 }},
              data: filtered,
            }};
          }});

          const robotOption = {{
            tooltip: {{
              trigger: 'item',
              formatter: function(p) {{
                const d = p.data.value;
                const dur = (d[2] - d[1]).toFixed(4);
                return '<b>Robot ' + d[0] + '</b><br/>'
                  + 'State: ' + (robotNameMap[d[3]] || d[3]) + '<br/>'
                  + 'Round: ' + d[4] + '<br/>'
                  + 'Start: ' + d[1].toFixed(4) + ' s<br/>'
                  + 'End: ' + d[2].toFixed(4) + ' s<br/>'
                  + 'Duration: ' + dur + ' s';
              }}
            }},
            legend: {{
              data: robotStateDefs.map(s => s.name),
              orient: 'horizontal',
              left: 'center',
              top: 0,
              textStyle: {{ fontSize: 11 }},
            }},
            grid: {{ left: 60, right: 30, top: 35, bottom: 30 }},
            xAxis: {{ type: 'value', name: 'Time (s)', nameLocation: 'center', nameGap: 25,
                      axisLabel: {{ formatter: v => v.toFixed(1) }} }},
            yAxis: {{ type: 'category', name: 'Robot', data: robots.map(r => 'Robot ' + r),
                      inverse: true }},
            series: robotSeries,
          }};
          chart.setOption(robotOption);
          window.addEventListener('resize', () => chart.resize());
        }})();

        // --- GPU Utilization pie chart ---
        (function() {{
          const chart = echarts.init(document.getElementById('utilPie'));
          const pieData = stats.gpu_utils.map(g => ({{
            name: 'GPU ' + g.gpu_id + ' Busy',
            value: parseFloat(g.busy_time.toFixed(4)),
          }}));
          // Add idle time for each GPU
          stats.gpu_utils.forEach(g => {{
            const idleTime = stats.total_time - g.busy_time;
            pieData.push({{
              name: 'GPU ' + g.gpu_id + ' Idle',
              value: parseFloat(Math.max(0, idleTime).toFixed(4)),
            }});
          }});

          const option = {{
            tooltip: {{ formatter: function(p) {{ return p.name + ': ' + p.value.toFixed(4) + ' s (' + p.percent + '%)'; }} }},
            legend: {{ orient: 'vertical', left: 10, top: 20, textStyle: {{ fontSize: 10 }} }},
            series: [{{
              type: 'pie',
              radius: ['30%', '65%'],
              center: ['60%', '55%'],
              label: {{ formatter: '{{b}}\\n{{d}}%', fontSize: 10 }},
              data: pieData,
              emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' }} }}
            }}]
          }};
          chart.setOption(option);
          window.addEventListener('resize', () => chart.resize());
        }})();

        // --- Wait time bar chart ---
        (function() {{
          const chart = echarts.init(document.getElementById('waitBar'));
          const option = {{
            tooltip: {{ valueFormatter: v => v.toFixed(4) + ' s' }},
            xAxis: {{ type: 'category', data: ['P50', 'P90', 'P99', 'Mean', 'Min', 'Max'] }},
            yAxis: {{ type: 'value', name: 'Wait Time (s)', nameLocation: 'center', nameGap: 35 }},
            series: [{{
              type: 'bar',
              data: [
                {{ value: stats.wait_p50, itemStyle: {{ color: '#5470c6' }} }},
                {{ value: stats.wait_p90, itemStyle: {{ color: '#91cc75' }} }},
                {{ value: stats.wait_p99, itemStyle: {{ color: '#fac858' }} }},
                {{ value: stats.wait_mean, itemStyle: {{ color: '#ee6666' }} }},
                {{ value: stats.wait_min, itemStyle: {{ color: '#73c0de' }} }},
                {{ value: stats.wait_max, itemStyle: {{ color: '#fc8452' }} }},
              ],
              label: {{ show: true, position: 'top', formatter: p => p.value.toFixed(4) + ' s', fontSize: 10 }},
            }}]
          }};
          chart.setOption(option);
          window.addEventListener('resize', () => chart.resize());
        }})();
        </script>
        </body>
        </html>
        """)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GPU-robot dynamic batching simulator"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4, help="Number of GPUs (default: 4)"
    )
    parser.add_argument(
        "--num_robots", type=int, default=8, help="Number of robots (default: 8)"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum batch size (must be even, <= 32; default: 16)",
    )
    parser.add_argument(
        "--idle_threshold",
        type=float,
        default=0.010,
        help="Max idle time to wait for more observations, in seconds (default: 0.010)",
    )
    parser.add_argument(
        "--network_latency",
        type=float,
        default=0.005,
        help="Network latency for obs upload and action download, in seconds (default: 0.005)",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=10, help="Number of rounds (default: 10)"
    )
    parser.add_argument(
        "--action_min",
        type=float,
        default=0.035,
        help="Min action execution time in seconds (default: 0.035)",
    )
    parser.add_argument(
        "--action_max",
        type=float,
        default=0.045,
        help="Max action execution time in seconds (default: 0.045)",
    )
    parser.add_argument(
        "--actions_per_block",
        type=int,
        default=30,
        help="Actions per block (default: 30)",
    )
    parser.add_argument(
        "--csv", type=str, default="", help="Path to speed_pi05.csv (auto-detected if omitted)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dynamic_batching_sim_report.html",
        help="Output HTML path (default: dynamic_batching_sim_report.html)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = SimConfig(
        num_gpus=args.num_gpus,
        num_robots=args.num_robots,
        max_batch_size=args.max_batch_size,
        idle_time_threshold=args.idle_threshold,
        network_latency=args.network_latency,
        num_rounds=args.num_rounds,
        action_time_min=args.action_min,
        action_time_max=args.action_max,
        actions_per_block=args.actions_per_block,
        csv_path=args.csv,
    )

    print("Configuration:")
    print(f"  GPUs: {config.num_gpus}, Robots: {config.num_robots}")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  Idle threshold: {config.idle_time_threshold*1000:.1f} ms")
    print(f"  Network latency: {config.network_latency*1000:.1f} ms")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Action time: {config.action_time_min*1000:.0f}–{config.action_time_max*1000:.0f} ms")
    print(f"  Actions per block: {config.actions_per_block}")
    print()

    sim = BatchingSimulator(config)
    stats = sim.run()

    print("Results:")
    print(f"  Total simulation time: {stats['total_time']:.3f} s")
    print(f"  Inferences run: {stats['num_inferences']}")
    print(f"  Observations processed: {stats['total_obs_processed']}")
    print(f"  Overall GPU utilization: {stats['overall_utilization_pct']:.1f}%")
    print(f"  Per-GPU utilization:")
    for g in stats["gpu_utils"]:
        print(f"    GPU {g['gpu_id']}: {g['utilization_pct']:.1f}% ({g['busy_time']:.3f} s busy)")
    print(f"  Robot wait time (s):")
    print(f"    P50={stats['wait_p50']:.4f}  P90={stats['wait_p90']:.4f}  P99={stats['wait_p99']:.4f}")
    print(f"    Mean={stats['wait_mean']:.4f}  Min={stats['wait_min']:.4f}  Max={stats['wait_max']:.4f}")
    print()

    sim.generate_html(stats, args.output)


if __name__ == "__main__":
    main()
