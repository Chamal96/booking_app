# --- Hotel booking module ---
import json
import os
import shutil
import tempfile
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.main import load_bookings, load_hotels, llm, save_to_faiss, BOOKINGS_PATH


def save_booking(booking_info):
    """
    Append a new booking to the booking list and safely save it to disk.

    Args:
        booking_info (dict): Structured booking data to be saved.
    """
    bookings = load_bookings()
    bookings.append(booking_info)

    os.makedirs(os.path.dirname(BOOKINGS_PATH), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(BOOKINGS_PATH), suffix=".json") as tmp_file:
        json.dump(bookings, tmp_file, indent=2)
        temp_path = tmp_file.name

    shutil.move(temp_path, BOOKINGS_PATH)


def book_hotel(user_input, memory):
    """
    Extract structured hotel booking info from user input using an LLM,
    fallback to basic extraction if parsing fails, and save the result.

    Args:
        user_input (str): User's natural language booking request.
        memory (ConversationBufferMemory): LangChain memory for context.

    Returns:
        str: A confirmation message with the booked hotel name.
    """
    hotels = load_hotels()
    hotel_names = [h['name'] for h in hotels]

    prompt = PromptTemplate.from_template("""
    You are a hotel booking assistant.
    {chat_history}
    User said: {user_input}

    Extract structured booking data as key-value pairs. If anything is missing, put 'Not Provided'. Return JSON format:
    {
      "hotel_name": "...",
      "guest_name": "...",
      "check_in_date": "...",
      "nights": "...",
      "user_request": "..."
    }
    """)

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    raw_response = chain.run(user_input=user_input)

    try:
        booking_data = json.loads(raw_response)
    except Exception:
        # Fallback to minimal extraction
        booked_hotel = next((name for name in hotel_names if name.lower() in user_input.lower()), "Unknown")
        booking_data = {
            "hotel_name": booked_hotel,
            "guest_name": "Not Provided",
            "check_in_date": "Not Provided",
            "nights": "Not Provided",
            "user_request": user_input
        }

    save_booking(booking_data)

    # Save booking query to FAISS for semantic retrieval
    save_to_faiss(
        text=booking_data["user_request"],
        metadata_entry=booking_data,
        entry_type="booking"
    )

    return f"✅ Booking confirmed for **{booking_data['hotel_name']}**."
