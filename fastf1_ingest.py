"""
fastf1_ingest.py
 
Fetches F1 race data via FastF1 and converts it into LangChain Document
objects (text + metadata) ready to be embedded and stored in Chroma.
 
This module knows nothing about LangChain's vector store, embeddings, or
Chroma — it only handles: fetch from FastF1 -> structured dicts -> Documents.
Keeping this separation means you can test FastF1 calls independently of
the RAG pipeline.
"""
 
import fastf1
from langchain_core.documents import Document
 
 
def get_race_results(year: int, race: str) -> list[dict]:
    """Fetch race results for a given season + race name/round.
 
    race: can be a race name ("Monaco"), round number, or country name —
    anything fastf1.get_session accepts.
    """
    session = fastf1.get_session(year, race, "R")  # "R" = Race session
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
 
 
def get_race_documents(year: int, race: str) -> list[Document]:
    """Convenience wrapper: fetch + convert in one call."""
    results = get_race_results(year, race)
    return race_results_to_documents(results, year, race)
 
 
if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity check ingestion
    # before wiring it into rag_pipeline.py
    docs = get_race_documents(2023, "Monaco")
    print(f"Generated {len(docs)} documents\n")
    for d in docs[:3]:
        print(d.page_content)
        print(d.metadata)
        print()
 