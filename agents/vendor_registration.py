# --- Hotel registration ---
from agents.main import load_hotels, save_hotels, save_to_faiss


def register_hotel(name, location, description, price_per_night, amenities):
    """
    Register a new hotel with provided details, save it to storage, and index it in FAISS.

    Args:
        name (str): Hotel name.
        location (str): Hotel location.
        description (str): Description of the hotel.
        price_per_night (str or float): Price per night.
        amenities (str): Comma-separated list of amenities.

    Returns:
        str: Confirmation message after successful registration.
    """
    hotel = {
        "name": name,
        "location": location,
        "description": description,
        "price_per_night": price_per_night,
        "amenities": amenities
    }

    # Load existing hotels and append the new one
    hotels = load_hotels()
    hotels.append(hotel)
    save_hotels(hotels)

    # Embed and store in FAISS for semantic search
    text = f"{name}. {location}. {description}. {price_per_night}. {amenities}"
    save_to_faiss(text, hotel, entry_type="registration")

    return "✅ Hotel registered successfully!"
