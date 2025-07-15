import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

# --- Configuration ---
API_KEY        = os.getenv("API_KEY")
MODEL          = os.getenv("MODEL")
DATA_PATH      = os.getenv("DATA_PATH")         # Path to hotels JSON
BOOKINGS_PATH  = os.getenv("BOOKINGS_PATH")     # Path to bookings JSON
INDEX_PATH     = os.getenv("INDEX_PATH", "faiss.index")
META_PATH      = INDEX_PATH + ".meta"

# --- Initialize Embedding Model & LLM ---
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()
llm = ChatOpenAI(model=MODEL, temperature=0.4, api_key=API_KEY)

# --- Load or Initialize FAISS Index & Metadata ---
if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []
else:
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    metadata = []

# --- Helper Functions ---

def load_hotels():
    """
    Load the list of hotels from the DATA_PATH JSON file.

    Returns:
        list: A list of hotel records.
    """
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return []

def save_hotels(hotels):
    """
    Save the list of hotels to the DATA_PATH JSON file.

    Args:
        hotels (list): List of hotel records to save.
    """
    with open(DATA_PATH, "w") as f:
        json.dump(hotels, f, indent=2)

def load_bookings():
    """
    Load the list of bookings from the BOOKINGS_PATH JSON file.

    Returns:
        list: A list of booking records.
    """
    if os.path.exists(BOOKINGS_PATH):
        with open(BOOKINGS_PATH, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return []

def save_bookings(bookings):
    """
    Save the list of bookings to the BOOKINGS_PATH JSON file.

    Args:
        bookings (list): List of booking records to save.
    """
    with open(BOOKINGS_PATH, "w") as f:
        json.dump(bookings, f, indent=2)

def save_to_faiss(text, metadata_entry, entry_type):
    """
    Encode text and save the embedding to the FAISS index with its metadata.

    Args:
        text (str): Text to embed and store.
        metadata_entry (dict): Associated metadata to store.
        entry_type (str): Type of entry ('hotel' or 'booking').
    """
    vector = EMBED_MODEL.encode([text]).astype(np.float32)
    faiss_index.add(vector)
    metadata.append({
        "type": entry_type,
        "vector_text": text,
        "data": metadata_entry
    })
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
