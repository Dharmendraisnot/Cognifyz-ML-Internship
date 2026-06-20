import streamlit as st
import pandas as pd
import pickle


with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="⭐",
    layout="centered"
)

st.title("🍽️ Restaurant Rating Prediction System")
st.write("Predict Restaurant Ratings using Machine Learning")

st.markdown("---")

votes = st.number_input(
    "Number of Votes",
    min_value=0,
    value=100
)

cost = st.number_input(
    "Average Cost for Two",
    min_value=0,
    value=500
)

price_range = st.selectbox(
    "Price Range",
    [1, 2, 3, 4]
)

table_booking = st.selectbox(
    "Has Table Booking",
    ["No", "Yes"]
)

online_delivery = st.selectbox(
    "Has Online Delivery",
    ["No", "Yes"]
)


table_booking = 1 if table_booking == "Yes" else 0
online_delivery = 1 if online_delivery == "Yes" else 0

if st.button("Predict Rating"):

    input_data = pd.DataFrame({
        'City':[0],
        'Cuisines':[0],
        'Average Cost for two':[cost],
        'Price range':[price_range],
        'Votes':[votes],
        'Has Table booking':[table_booking],
        'Has Online delivery':[online_delivery]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Rating: {prediction[0]:.2f} ⭐"
    )
st.image(
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
    use_container_width=True
)    