from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# Sample documents taken from fastf1_ingest.get_race_documents() output
# (2023 Monaco + 2023 Australia), so the schema matches the real pipeline.
#
# Metadata schema:
#   all documents:  doc_id, season, race, type
#   race_result:    driver (full name), team, position
#   events:         round, session ("R"), lap, drivers ("VER,HAM"),
#                   t_start / t_end (FastF1 session time in seconds, used to
#                   seek the race replay)
#   type is one of: race_result, overtake, pit_stop, track_status, race_control
documents = [
    # --- race_result: final classification, no timestamp ---
    Document(
        page_content="In the 2023 Monaco Grand Prix, Max Verstappen (Red Bull Racing) finished in position 1, after starting from grid position 1, scoring 25.0 championship points.",
        metadata={"doc_id": "2023-Monaco-R-result-VER", "season": 2023, "race": "Monaco", "driver": "Max Verstappen", "team": "Red Bull Racing", "position": 1.0, "type": "race_result"},
    ),
    Document(
        page_content="In the 2023 Monaco Grand Prix, Fernando Alonso (Aston Martin) finished in position 2, after starting from grid position 2, scoring 18.0 championship points.",
        metadata={"doc_id": "2023-Monaco-R-result-ALO", "season": 2023, "race": "Monaco", "driver": "Fernando Alonso", "team": "Aston Martin", "position": 2.0, "type": "race_result"},
    ),
    Document(
        page_content="In the 2023 Monaco Grand Prix, Esteban Ocon (Alpine) finished in position 3, after starting from grid position 3, scoring 15.0 championship points.",
        metadata={"doc_id": "2023-Monaco-R-result-OCO", "season": 2023, "race": "Monaco", "driver": "Esteban Ocon", "team": "Alpine", "position": 3.0, "type": "race_result"},
    ),
    Document(
        page_content="In the 2023 Monaco Grand Prix, Lance Stroll (Aston Martin) did not classify (result: R), after starting from grid position 14, with a race status of 'Retired', scoring 0.0 championship points.",
        metadata={"doc_id": "2023-Monaco-R-result-STR", "season": 2023, "race": "Monaco", "driver": "Lance Stroll", "team": "Aston Martin", "position": 20.0, "type": "race_result"},
    ),

    # --- overtake: window covers the overtaking driver's lap ---
    Document(
        page_content="In the 2023 Monaco Grand Prix, on lap 17, Kevin Magnussen (Haas F1 Team) overtook Logan Sargeant (Williams) to move up to P15 (MAG on hard tyres, SAR on medium tyres).",
        metadata={"doc_id": "2023-Monaco-R-overtake-17-MAG-SAR", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "overtake", "lap": 17, "t_start": 5011.392, "t_end": 5110.598, "drivers": "MAG,SAR"},
    ),
    Document(
        page_content="In the 2023 Australia Grand Prix, on lap 12, Max Verstappen (Red Bull Racing) overtook Lewis Hamilton (Mercedes) to move up to P1 (VER on hard tyres, HAM on hard tyres).",
        metadata={"doc_id": "2023-Australia-R-overtake-12-VER-HAM", "season": 2023, "race": "Australia", "round": 3, "session": "R", "type": "overtake", "lap": 12, "t_start": 5909.463, "t_end": 6011.457, "drivers": "VER,HAM"},
    ),

    # --- pit_stop: pit entry -> pit exit ---
    Document(
        page_content="In the 2023 Monaco Grand Prix, Lewis Hamilton (Mercedes) pitted on lap 31, switching from medium to hard tyres (24.5s in the pit lane), running P8 before the stop and P8 after it.",
        metadata={"doc_id": "2023-Monaco-R-pit_stop-31-HAM", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "pit_stop", "lap": 31, "t_start": 6144.329, "t_end": 6188.797, "drivers": "HAM"},
    ),
    Document(
        page_content="In the 2023 Monaco Grand Prix, Fernando Alonso (Aston Martin) pitted on lap 54, switching from hard to medium tyres (24.2s in the pit lane), running P2 before the stop and P2 after it.",
        metadata={"doc_id": "2023-Monaco-R-pit_stop-54-ALO", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "pit_stop", "lap": 54, "t_start": 7941.906, "t_end": 7986.125, "drivers": "ALO"},
    ),
    Document(
        page_content="In the 2023 Monaco Grand Prix, Lewis Hamilton (Mercedes) pitted on lap 54, switching from hard to intermediate tyres (26.6s in the pit lane), running P7 before the stop and P4 after it.",
        metadata={"doc_id": "2023-Monaco-R-pit_stop-54-HAM", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "pit_stop", "lap": 54, "t_start": 7978.562, "t_end": 8025.202, "drivers": "HAM"},
    ),

    # --- race_control: stewards' decisions and flags ---
    Document(
        page_content='In the 2023 Monaco Grand Prix, on lap 5, race control announced: "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 27 (HUL) - CAUSING A COLLISION". Drivers involved: Nico Hulkenberg (Haas F1 Team).',
        metadata={"doc_id": "2023-Monaco-R-race_control-21", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "race_control", "lap": 5, "t_start": 4029.177, "t_end": 4069.177, "drivers": "HUL"},
    ),
    Document(
        page_content='In the 2023 Monaco Grand Prix, on lap 59, race control announced: "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 2 (SAR) - SPEEDING IN THE PIT LANE". Drivers involved: Logan Sargeant (Williams).',
        metadata={"doc_id": "2023-Monaco-R-race_control-153", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "race_control", "lap": 59, "t_start": 8435.177, "t_end": 8475.177, "drivers": "SAR"},
    ),
    Document(
        page_content='In the 2023 Monaco Grand Prix, on lap 17, race control announced: "BLACK AND WHITE FLAG FOR CAR 55 (SAI) - CAUSING A COLLISION". Drivers involved: Carlos Sainz (Ferrari).',
        metadata={"doc_id": "2023-Monaco-R-race_control-31", "season": 2023, "race": "Monaco", "round": 6, "session": "R", "type": "race_control", "lap": 17, "t_start": 4940.177, "t_end": 4980.177, "drivers": "SAI"},
    ),

    # --- track_status: Safety Car / VSC / red flag periods ---
    Document(
        page_content="In the 2023 Australia Grand Prix, the Safety Car was deployed on lap 7 and ended on lap 8, lasting 158 seconds. Drivers who pitted under it: Kevin Magnussen (Haas F1 Team), George Russell (Mercedes), Carlos Sainz (Ferrari).",
        metadata={"doc_id": "2023-Australia-R-track_status-7-SafetyCar", "season": 2023, "race": "Australia", "round": 3, "session": "R", "type": "track_status", "lap": 7, "t_start": 4393.246, "t_end": 4571.056, "drivers": "MAG,RUS,SAI"},
    ),
    Document(
        page_content="In the 2023 Australia Grand Prix, the Red Flag was deployed on lap 8 and ended on lap 8, lasting 941 seconds.",
        metadata={"doc_id": "2023-Australia-R-track_status-8-RedFlag", "season": 2023, "race": "Australia", "round": 3, "session": "R", "type": "track_status", "lap": 8, "t_start": 4551.056, "t_end": 5511.671, "drivers": ""},
    ),
    Document(
        page_content="In the 2023 Australia Grand Prix, the Virtual Safety Car was deployed on lap 18 and ended on lap 19, lasting 137 seconds.",
        metadata={"doc_id": "2023-Australia-R-track_status-18-VirtualSafetyCar", "season": 2023, "race": "Australia", "round": 3, "session": "R", "type": "track_status", "lap": 18, "t_start": 6444.722, "t_end": 6601.267, "drivers": ""},
    ),
]


print(f"Loaded {len(documents)} documents for the RAG demo.")

vectorstore = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name="fastf1_rag_demo",
)

#Create retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3} #Return top 3
)

BM25_retriever = BM25Retriever.from_documents(
    documents,
    k=3 #Return top 3
)

#Combine with ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, BM25_retriever],
    weights=[0.5, 0.5] #Equal weight for both retrievers
)

print("Ensemble retriever created.")