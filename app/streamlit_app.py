import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Movie Recommendation System")

st.write("Enter a user ID to get personalized movie recommendations.")

# user input
user_id = st.number_input("User ID",min_value=1,value=149954)

num_recommendations = st.slider("Number of recommendations",5,20,10)

if st.button("Get Recommendations"):

    url = f"{API_URL}/recommend/{user_id}?n={num_recommendations}"

    response = requests.get(url)

    if response.status_code == 200:

        data = response.json()

        st.subheader(f"Recommendations for user {user_id}")

        for i, rec in enumerate(data["recommendations"],1):
            st.write(f"{i}. {rec['movie']}(score:{rec['score']:.3f})")
    else:
        st.error("Failed to get recommendations")