"""
fastf1_ingest.py

Fetches F1 race data via FastF1 and converts it into LangChain Document
objects (text + metadata) ready to be embedded and stored in Chroma.

This module knows nothing about LangChain's vector store, embeddings, or
Chroma — it only handles: fetch from FastF1 -> structured dicts -> Documents.
Keeping this separation means you can test FastF1 calls independently of
the RAG pipeline.

Two kinds of documents are produced:
  - race_result: one per driver, final classification (no timestamp)
  - events: overtake, pit_stop, track_status (SC/VSC/red flag) and
    race_control messages. Each carries replay metadata:
        lap      -> lap number the event happened on
        t_start  -> session time (seconds) where the replay should start
        t_end    -> session time (seconds) where the replay should stop
        drivers  -> comma-separated driver abbreviations involved ("VER,ALO")
    "Session time" is FastF1's clock (laps["Time"], telemetry["SessionTime"]),
    so it can be used to seek directly in anything built on FastF1 telemetry.
"""

import os
import re

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError
from langchain_core.documents import Document

os.makedirs('./fastf1_cache', exist_ok=True)
fastf1.Cache.enable_cache('./fastf1_cache')

# Seconds of lead-in / lead-out around an event so the replay shows the build-up.
REPLAY_PADDING = 10.0

# Race control flags that are too frequent/routine to be worth embedding.
IGNORED_FLAGS = {"BLUE", "CLEAR", "YELLOW"}

# FastF1 track status codes worth turning into events.
TRACK_STATUS_EVENTS = {
    "4": "Safety Car",
    "5": "Red Flag",
    "6": "Virtual Safety Car",
}


def load_race_session(year: int, race: str, telemetry: bool = False):
    """Load a race session with laps and race control messages.

    race: can be a race name ("Monaco"), round number, or country name —
    anything fastf1.get_session accepts.

    telemetry=False keeps ingestion fast; event timestamps are still accurate
    to ~2 seconds (see _session_time_offset).
    """
    session = fastf1.get_session(year, race, "R")  # "R" = Race session
    session.load(laps=True, telemetry=telemetry, weather=False, messages=True)
    return session


def get_race_results(year: int, race: str) -> list[dict]:
    """Fetch race results for a given season + race name/round."""
    session = fastf1.get_session(year, race, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session.results.to_dict("records")


def race_results_to_documents(results: list[dict], year: int, race_name: str) -> list[Document]:
    """Convert raw FastF1 result rows into Document objects with metadata."""
    docs = []

    for row in results:
        position = row.get("Position")
        classified = row.get("ClassifiedPosition")
        grid = row.get("GridPosition")
        status = row.get("Status")
        points = row.get("Points")
        driver = row.get("FullName")
        team = row.get("TeamName")

        # Build a natural-language sentence from the structured row.
        # This is the "translation" step that makes tabular data
        # searchable via semantic embedding.
        if classified and classified.isdigit():
            finish_phrase = f"finished in position {classified}"
        else:
            # classified is something like "DNF", "DSQ", "DNS"
            finish_phrase = f"did not classify (result: {classified})"

        sentence_parts = [
            f"In the {year} {race_name} Grand Prix, {driver} ({team}) {finish_phrase}"
        ]

        if grid is not None:
            sentence_parts.append(f"after starting from grid position {int(grid)}")

        if status and status != "Finished":
            sentence_parts.append(f"with a race status of '{status}'")

        if points is not None:
            sentence_parts.append(f"scoring {points} championship points")

        text = ", ".join(sentence_parts) + "."

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": f"{year}-{race_name}-R-result-{row.get('Abbreviation')}",
                    "season": year,
                    "race": race_name,
                    "driver": driver,
                    "team": team,
                    "position": position,
                    "type": "race_result",
                },
            )
        )

    return docs


# ---------------------------------------------------------------------------
# Event ingestion
# ---------------------------------------------------------------------------

def _seconds(td) -> float | None:
    """Timedelta -> float seconds (None for NaT)."""
    if pd.isna(td):
        return None
    return round(td.total_seconds(), 3)


def _session_time_offset(session) -> pd.Timestamp:
    """Wall-clock datetime of session time zero.

    Race control messages are timestamped with wall-clock datetimes while
    laps/track status use session time. FastF1 exposes the offset as
    session.t0_date, but only when telemetry is loaded. Otherwise, calibrate
    it from the chequered flag, which is shown as the winner crosses the line
    (accurate to ~2s, the resolution of race control timestamps).
    """
    try:
        return session.t0_date
    except DataNotLoadedError:
        pass

    msgs = session.race_control_messages
    chequered = msgs[msgs["Flag"] == "CHEQUERED"]
    winner = session.results.iloc[0]["Abbreviation"]
    winner_finish = session.laps.pick_drivers(winner)["Time"].max()

    if chequered.empty or pd.isna(winner_finish):
        raise ValueError(
            "Cannot align race control messages to session time; "
            "load the session with telemetry=True."
        )
    return chequered["Time"].iloc[0] - winner_finish


def _lap_at(laps: pd.DataFrame, t: float) -> int:
    """Lap the race leader was on at session time t (seconds)."""
    started = laps[laps["LapStartTime"].dt.total_seconds() <= t]
    if started.empty:
        return 1
    return int(started["LapNumber"].max())


class _RaceContext:
    """Shared lookups for building event documents from one session."""

    def __init__(self, session, race_name: str):
        self.session = session
        self.year = session.event.year
        self.round = int(session.event["RoundNumber"])
        self.race_name = race_name
        self.laps = session.laps

        results = session.results
        self.full_name = dict(zip(results["Abbreviation"], results["FullName"]))
        self.team = dict(zip(results["Abbreviation"], results["TeamName"]))
        self.abbr_by_number = dict(zip(results["DriverNumber"].astype(str), results["Abbreviation"]))

    def who(self, abbr: str) -> str:
        """'VER' -> 'Max Verstappen (Red Bull Racing)'."""
        return f"{self.full_name.get(abbr, abbr)} ({self.team.get(abbr, 'unknown team')})"

    def document(self, event_type: str, key: str, text: str, lap: int,
                 t_start: float, t_end: float, drivers: list[str]) -> Document:
        return Document(
            page_content=f"In the {self.year} {self.race_name} Grand Prix, {text}",
            metadata={
                "doc_id": f"{self.year}-{self.race_name}-R-{event_type}-{key}",
                "season": self.year,
                "race": self.race_name,
                "round": self.round,
                "session": "R",
                "type": event_type,
                "lap": int(lap),
                "t_start": round(max(t_start - REPLAY_PADDING, 0.0), 3),
                "t_end": round(t_end + REPLAY_PADDING, 3),
                "drivers": ",".join(drivers),
            },
        )


def _overtake_documents(ctx: _RaceContext) -> list[Document]:
    """Detect on-track position swaps by comparing positions at the end of
    consecutive laps. Swaps where either driver pitted are excluded, since
    those are pit-cycle position changes rather than overtakes.

    The window covers the overtaking driver's whole lap; narrowing it to the
    exact moment needs telemetry (distance/position data).
    """
    laps = ctx.laps[ctx.laps["Position"].notna()]
    by_lap = {n: g.set_index("Driver") for n, g in laps.groupby("LapNumber")}
    docs = []

    for lap_number in sorted(by_lap):
        prev = by_lap.get(lap_number - 1)
        if prev is None:
            continue  # lap 1: start-line chaos, not comparable to a previous lap
        cur = by_lap[lap_number]

        drivers = cur.index.intersection(prev.index)
        pitted = {
            d for d in drivers
            if pd.notna(cur.at[d, "PitInTime"]) or pd.notna(cur.at[d, "PitOutTime"])
            or pd.notna(prev.at[d, "PitInTime"])
        }

        for attacker in drivers:
            for defender in drivers:
                if attacker == defender or attacker in pitted or defender in pitted:
                    continue
                was_behind = prev.at[attacker, "Position"] > prev.at[defender, "Position"]
                now_ahead = cur.at[attacker, "Position"] < cur.at[defender, "Position"]
                if not (was_behind and now_ahead):
                    continue

                row = cur.loc[attacker]
                new_pos = int(row["Position"])
                text = (
                    f"on lap {int(lap_number)}, {ctx.who(attacker)} overtook "
                    f"{ctx.who(defender)} to move up to P{new_pos}"
                )
                if isinstance(row["Compound"], str) and isinstance(cur.at[defender, "Compound"], str):
                    text += (
                        f" ({attacker} on {row['Compound'].lower()} tyres, "
                        f"{defender} on {cur.at[defender, 'Compound'].lower()} tyres)"
                    )
                docs.append(ctx.document(
                    "overtake", f"{int(lap_number)}-{attacker}-{defender}", text + ".",
                    lap=lap_number,
                    t_start=_seconds(row["LapStartTime"]),
                    t_end=_seconds(row["Time"]),
                    drivers=[attacker, defender],
                ))

    return docs


def _pit_stop_documents(ctx: _RaceContext) -> list[Document]:
    """One document per pit entry: in-lap -> out-lap, with tyre change."""
    docs = []

    for driver, driver_laps in ctx.laps.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)

        for i, in_lap in driver_laps[driver_laps["PitInTime"].notna()].iterrows():
            lap_number = int(in_lap["LapNumber"])
            pit_in = _seconds(in_lap["PitInTime"])
            out_lap = driver_laps.iloc[i + 1] if i + 1 < len(driver_laps) else None
            pit_out = _seconds(out_lap["PitOutTime"]) if out_lap is not None else None

            if pit_out is None:
                text = f"{ctx.who(driver)} entered the pits on lap {lap_number} and retired."
                t_end = pit_in + 30
            else:
                old_tyre = in_lap["Compound"]
                new_tyre = out_lap["Compound"]
                text = (
                    f"{ctx.who(driver)} pitted on lap {lap_number}, "
                    f"switching from {str(old_tyre).lower()} to {str(new_tyre).lower()} tyres "
                    f"({pit_out - pit_in:.1f}s in the pit lane)"
                )
                if pd.notna(in_lap["Position"]) and pd.notna(out_lap["Position"]):
                    text += (
                        f", running P{int(in_lap['Position'])} before the stop "
                        f"and P{int(out_lap['Position'])} after it"
                    )
                text += "."
                t_end = pit_out

            docs.append(ctx.document(
                "pit_stop", f"{lap_number}-{driver}", text,
                lap=lap_number, t_start=pit_in, t_end=t_end, drivers=[driver],
            ))

    return docs


def _track_status_documents(ctx: _RaceContext) -> list[Document]:
    """Safety Car / VSC / red flag periods, from deployment to all clear."""
    status = ctx.session.track_status.sort_values("Time")
    race_end = _seconds(ctx.laps["Time"].max())
    periods = []
    current = None

    for _, row in status.iterrows():
        code, t = str(row["Status"]), _seconds(row["Time"])
        if t > race_end:
            break  # post-race statuses (e.g. red flag after the chequered flag)

        kind = TRACK_STATUS_EVENTS.get(code)
        if current is not None and (code == "1" or (kind and kind != current["kind"])):
            # AllClear, or escalation such as Safety Car -> Red Flag
            current["end"] = t
            periods.append(current)
            current = None
        if current is None and kind:
            current = {"kind": kind, "start": t}
    if current is not None:
        current["end"] = race_end
        periods.append(current)

    docs = []
    for period in periods:
        start, end = period["start"], period["end"]
        start_lap, end_lap = _lap_at(ctx.laps, start), _lap_at(ctx.laps, end)

        pit_in = ctx.laps["PitInTime"].dt.total_seconds()
        pitted = sorted(ctx.laps[(pit_in >= start) & (pit_in <= end)]["Driver"].unique())

        text = (
            f"the {period['kind']} was deployed on lap {start_lap} and ended on lap {end_lap}, "
            f"lasting {end - start:.0f} seconds."
        )
        if period["kind"] == "Red Flag":
            pitted = []  # every car returns to the pit lane under a red flag
        if pitted:
            text += " Drivers who pitted under it: " + ", ".join(ctx.who(d) for d in pitted) + "."

        docs.append(ctx.document(
            "track_status", f"{start_lap}-{period['kind'].replace(' ', '')}", text,
            lap=start_lap, t_start=start, t_end=end, drivers=pitted,
        ))

    return docs


def _race_control_documents(ctx: _RaceContext) -> list[Document]:
    """Stewards' decisions, incidents, black-and-white flags, etc.

    The replay window leads into the message, since race control usually
    reports an incident shortly after it happens.
    """
    offset = _session_time_offset(ctx.session)
    msgs = ctx.session.race_control_messages
    msgs = msgs[msgs["Category"] != "Drs"]
    msgs = msgs[~msgs["Flag"].isin(IGNORED_FLAGS)]
    docs = []

    for i, msg in msgs.iterrows():
        t = _seconds(msg["Time"] - offset)
        message = str(msg["Message"]).strip()

        drivers = re.findall(r"\(([A-Z]{3})\)", message)
        if not drivers and msg["RacingNumber"]:
            number_driver = ctx.abbr_by_number.get(str(msg["RacingNumber"]))
            drivers = [number_driver] if number_driver else []
        drivers = list(dict.fromkeys(d for d in drivers if d in ctx.full_name))

        text = f"on lap {int(msg['Lap'])}, race control announced: \"{message}\"."
        if drivers:
            text += " Drivers involved: " + ", ".join(ctx.who(d) for d in drivers) + "."

        docs.append(ctx.document(
            "race_control", str(i), text,
            lap=msg["Lap"], t_start=t - 20, t_end=t, drivers=drivers,
        ))

    return docs


def race_events_to_documents(session, race_name: str) -> list[Document]:
    """Build timestamped event documents from a loaded race session."""
    ctx = _RaceContext(session, race_name)
    docs = (
        _overtake_documents(ctx)
        + _pit_stop_documents(ctx)
        + _track_status_documents(ctx)
        + _race_control_documents(ctx)
    )
    return sorted(docs, key=lambda d: d.metadata["t_start"])


def get_race_documents(year: int, race: str) -> list[Document]:
    """Convenience wrapper: fetch + convert results and events in one call."""
    session = load_race_session(year, race)
    results = race_results_to_documents(session.results.to_dict("records"), year, race)
    return results + race_events_to_documents(session, race)


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity check ingestion
    # before wiring it into rag_pipeline.py
    from collections import Counter

    docs = get_race_documents(2023, "Monaco")
    print(f"Generated {len(docs)} documents: {dict(Counter(d.metadata['type'] for d in docs))}\n")
    for event_type in ("overtake", "pit_stop", "track_status", "race_control"):
        for d in [d for d in docs if d.metadata["type"] == event_type][:2]:
            print(d.page_content)
            print(d.metadata)
            print()
