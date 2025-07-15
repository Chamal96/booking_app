import numpy as np
from agents.main import llm, EMBED_MODEL, faiss_index, metadata


def search_booked_hotels(user_query, top_k=5):
    """
    Search for relevant hotel bookings using a semantic vector search
    and generate a natural language response using the LLM.

    Args:
        user_query (str): Natural language query from the user.
        top_k (int): Number of top similar results to retrieve.

    Returns:
        str: LLM-generated answer based on retrieved booking data.
    """
    # 1. Convert user query into an embedding
    query_vector = EMBED_MODEL.encode([user_query]).astype(np.float32)

    # 2. Search the FAISS index for similar bookings
    _, indices = faiss_index.search(query_vector, top_k)

    # 3. Filter only metadata entries of type "booking"
    results = [
        metadata[idx]["data"]
        for idx in indices[0]
        if idx < len(metadata) and metadata[idx]["type"] == "booking"
    ]

    if not results:
        return "No relevant bookings found."

    # 4. Format results as context for the LLM
    context_text = "\n\n".join([
        f"Booking: Hotel={r['hotel_name']}, Guest={r['guest_name']}, "
        f"Check-in={r['check_in_date']}, Nights={r['nights']}"
        for r in results
    ])

    prompt_text = f"""
    You are a helpful assistant. Based on the following booking information, answer the user's question.

    Booking Data:
    {context_text}

    User question: {user_query}

    Answer:
    """

    # 5. Use LLM to generate a context-aware answer
    return llm.call_as_llm(prompt_text)
