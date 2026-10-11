"""
replay_launcher.py

Opens the f1-race-replay window (vendored in replay/) positioned at one
moment of a race: playback starts at --start, the --focus drivers are
highlighted, and playback pauses at --end.

--start / --end are FastF1 session times in seconds: the same t_start / t_end
values fastf1_ingest.py stores in event document metadata. The replay counts
its own time from the first telemetry sample, so they are converted with the
race's t0_offset (saved in the replay's computed_data cache).

Usage:
    python replay_launcher.py --year 2023 --round 6 --start 7978.5 --end 8025.2 --focus HAM

Keys (in addition to the replay's own controls):
    M   replay the moment again from --start
"""

import argparse
import logging
import os
import sys

# The replay's code imports itself as "src..." and reads/writes relative paths
# (computed_data/, .fastf1-cache), so it has to run from inside replay/.
# This runs at import time so multiprocessing workers (used when the cache is
# rebuilt) can import the replay's modules too.
REPLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay")
os.chdir(REPLAY_DIR)
sys.path.insert(0, REPLAY_DIR)

# The replay prints symbols like "✗"; when output is piped (e.g. launched as a
# subprocess), Windows defaults to cp1252 and those prints crash the replay.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import arcade

import main as replay_main
import src.run_session as run_session
from src.f1_data import FPS
from src.interfaces.race_replay import F1RaceReplayWindow


class MomentReplayWindow(F1RaceReplayWindow):
    """The replay window, opened at a moment instead of the race start."""

    def __init__(self, *args, start_s, end_s, focus, t0_offset, **kwargs):
        super().__init__(*args, **kwargs)
        last_frame = self.n_frames - 1

        self.start_frame = self._to_frame(start_s, t0_offset)
        self.end_frame = self._to_frame(end_s, t0_offset) if end_s is not None else None
        self.stopped_at_end = False
        self.frame_index = float(self.start_frame)

        if focus:
            # The leaderboard, driver boxes and car labels all read these each frame.
            self.selected_drivers = list(focus)
            self.selected_driver = focus[-1]

        end_text = f"frame {self.end_frame}" if self.end_frame is not None else "end of race"
        print(f"Moment: frame {self.start_frame} -> {end_text} of {last_frame} "
              f"(replay time {self.start_frame / FPS:.1f}s, t0_offset {t0_offset:.3f}s)")

    def _to_frame(self, session_s, t0_offset):
        """FastF1 session time (s) -> replay frame index, clamped to the race."""
        frame = int(round((session_s - t0_offset) * FPS))
        return min(max(frame, 0), self.n_frames - 1)

    def replay_moment(self):
        self.frame_index = float(self.start_frame)
        self.stopped_at_end = False
        self.paused = False

    def on_update(self, delta_time: float):
        super().on_update(delta_time)
        # Pause once at the end of the moment; after that the user can play on.
        if (self.end_frame is not None and not self.stopped_at_end
                and self.frame_index >= self.end_frame):
            self.frame_index = float(self.end_frame)
            self.paused = True
            self.stopped_at_end = True

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.M:
            self.replay_moment()
            return
        super().on_key_press(symbol, modifiers)


def parse_args():
    parser = argparse.ArgumentParser(description="Open the race replay at a specific moment.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, help="round number, e.g. 6 for 2023 Monaco")
    parser.add_argument("--start", type=float, required=True, help="FastF1 session time (s) to start at")
    parser.add_argument("--end", type=float, help="FastF1 session time (s) to pause at")
    parser.add_argument("--focus", default="", help="comma-separated driver codes to highlight, e.g. HAM,ALO")
    parser.add_argument("--insights", action="store_true", help="also open the replay's insights menu")
    parser.add_argument("--refresh-data", action="store_true",
                        help="rebuild the replay's telemetry cache (read by the replay itself)")
    return parser.parse_args()


def main():
    args = parse_args()
    focus = [code.strip().upper() for code in args.focus.split(",") if code.strip()]

    if "--verbose" not in sys.argv:
        logging.getLogger("fastf1").setLevel(logging.CRITICAL)

    # The replay's main() loads the race telemetry, then builds its window.
    # Capture the telemetry on the way past to read t0_offset from it...
    telemetry = {}
    load_race_telemetry = replay_main.get_race_telemetry

    def capture_race_telemetry(session, session_type="R"):
        data = load_race_telemetry(session, session_type=session_type)
        telemetry.update(data)
        return data

    replay_main.get_race_telemetry = capture_race_telemetry

    # ...and build MomentReplayWindow instead of the plain replay window.
    def build_window(*window_args, **window_kwargs):
        if "t0_offset" not in telemetry:
            sys.exit("This race's replay cache predates t0_offset. "
                     "Run again with --refresh-data to rebuild it (takes several minutes).")

        frame_drivers = telemetry["frames"][0]["drivers"]
        unknown = [code for code in focus if code not in frame_drivers]
        if unknown:
            print(f"Warning: no telemetry for {', '.join(unknown)}; "
                  f"drivers in this race: {', '.join(sorted(frame_drivers))}")

        return MomentReplayWindow(
            *window_args,
            start_s=args.start,
            end_s=args.end,
            focus=[code for code in focus if code in frame_drivers],
            t0_offset=telemetry["t0_offset"],
            **window_kwargs,
        )

    run_session.F1RaceReplayWindow = build_window

    if not args.insights:
        replay_main.launch_insights_menu = lambda: None

    replay_main.main(args.year, args.round)


if __name__ == "__main__":
    main()
