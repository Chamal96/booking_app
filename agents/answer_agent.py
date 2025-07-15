import json
import os
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from agents.main import DATA_PATH, INDEX_PATH, EMBED_MODEL, llm


def load_hotels():
    """
    Load hotel data from the local JSON file defined by DATA_PATH.

    Returns:
        list: List of hotel dictionaries.
    """
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return []


def load_index():
    """
    Load the FAISS index from disk or initialize a new index if not found.

    Returns:
        faiss.IndexFlatL2: The FAISS index object.
    """
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(384)  # Default embedding dimension for MiniLM


def embed_query(query):
    """
    Embed a textual query using the pre-loaded embedding model.

    Args:
        query (str): The user input query.

    Returns:
        np.ndarray: Query embedding vector of shape (1, dim).
    """
    return EMBED_MODEL.encode([query]).astype(np.float32)


def retrieve_relevant_hotels(query, top_k=5):
    """
    Retrieve the top-k most relevant hotels using FAISS similarity search.

    Args:
        query (str): User query string.
        top_k (int): Number of top results to return.

    Returns:
        list: List of hotel summary strings.
    """
    hotels = load_hotels()
    index = load_index()
    query_vec = embed_query(query)
    _, indices = index.search(query_vec, top_k)

    return [
        f"{hotels[idx]['name']} in {hotels[idx]['location']}. "
        f"Price: {hotels[idx]['price_per_night']}. "
        f"Description: {hotels[idx]['description']}. "
        f"Amenities: {hotels[idx]['amenities']}"
        for idx in indices[0] if idx < len(hotels)
    ]


def generate_answer(query):
    """
    Generate an LLM-based response to the user's hotel-related query
    using retrieved hotel information as context.

    Args:
        query (str): User query string.

    Returns:
        str: LLM-generated answer.
    """
    context = "\n".join(retrieve_relevant_hotels(query))

    prompt = PromptTemplate.from_template(
        "You are a helpful hotel‑booking assistant.\n"
        "{chat_history}\n"
        "Here are some hotel listings:\n\n{context}\n\n"
        "Answer the user’s question using **only** the above listings.\n\n"
        "Make well structured answer\n\n"
        "Question: {question}"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    return chain.run(context=context, question=query)
