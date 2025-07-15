import streamlit as st
from agents.vendor_registration import register_hotel
from agents.search_agent import search_booked_hotels
from agents.answer_agent import generate_answer
from agents.booking_agent import book_hotel
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Booking App", layout="centered")

st.markdown("<h1 style='text-align:center; color:#1f2937;'>🏨 Booking App</h1>", unsafe_allow_html=True)

tabs = st.tabs(["Vendor", "Inquery", "Booking", "Search Booked Hotels"])

### ------------------------ Vendor Registration ------------------------
with tabs[0]:
    st.markdown("### 🏨 Hotel Vendor Registration")

    with st.form("vendor_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Hotel Name", placeholder="Enter the hotel name")
        with col2:
            location = st.text_input("Location", placeholder="City or area")

        desc = st.text_area("Description", placeholder="Briefly describe the hotel", height=100)
        price = st.number_input("Price per Night (USD)", min_value=0.0, format="%.2f")
        amenities = st.text_input("Amenities (comma-separated)", placeholder="WiFi, Pool, Breakfast...")

        submitted = st.form_submit_button("Register Hotel")

        if submitted:
            status = register_hotel(name, location, desc, price, amenities)
            st.success(status)

### ------------------------ Search Hotels ------------------------
with tabs[3]:
    st.markdown("### 🔍 Search Booked Hotels")

    query = st.text_input("Your Query", placeholder="e.g., I am Maya, Let me see my booking.")
    if st.button("Search"):
        results = search_booked_hotels(query)
        print(results)
        if not results:
            st.warning("No hotels found for your query.")
        else:
            st.info(results)


### ------------------------ Ask Question ------------------------
with tabs[1]:
    st.markdown("### 💬 Ask a question about available hotels")

    question = st.text_input("Your Question", placeholder="E.g., Which hotels have free parking?")
    if st.button("Get Answer"):
        answer = generate_answer(question)
        st.info(answer)

### ------------------------ Booking Agent ------------------------
with tabs[2]:
    st.markdown("### 📝 Book a Hotel")

    with st.form("booking_form"):
        user_input = st.text_area(
            "Enter your booking request 👇",
            placeholder="e.g., I want to book City Lights Inn for 1 night starting July 11, 2025. My name is Chamal Jayasinghe. I’d like free Wi-Fi.",
            height=150,
        )

        confirm = st.form_submit_button("Confirm Booking")

        if confirm and user_input:
            memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
            response = book_hotel(user_input, memory)
            st.success(response)

