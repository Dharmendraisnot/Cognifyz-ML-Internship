import streamlit as st
import pandas as pd
import pickle


with open("cuisine_model.pkl", "rb") as file:
    model = pickle.load(file)


with open("city_encoder.pkl", "rb") as file:
    city_encoder = pickle.load(file)

with open("cuisine_encoder.pkl", "rb") as file:
    cuisine_encoder = pickle.load(file)

st.set_page_config(
    page_title="Cuisine Classification",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant Cuisine Classification")

st.write(
    "Predict the Cuisine Type of a Restaurant using Machine Learning"
)

st.markdown("---")


city_list = list(city_encoder.classes_)

city = st.selectbox(
    "Select City",
    city_list
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

votes = st.number_input(
    "Number of Votes",
    min_value=0,
    value=100
)

rating = st.slider(
    "Aggregate Rating",
    0.0,
    5.0,
    3.5
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

city_encoded = city_encoder.transform([city])[0]

if st.button("Predict Cuisine"):

    input_data = pd.DataFrame({
        'City':[city_encoded],
        'Average Cost for two':[cost],
        'Price range':[price_range],
        'Votes':[votes],
        'Aggregate rating':[rating],
        'Has Table booking':[table_booking],
        'Has Online delivery':[online_delivery]
    })

    prediction = model.predict(input_data)

    cuisine_name = cuisine_encoder.inverse_transform(prediction)

    st.success(
        f"Predicted Cuisine: {cuisine_name[0]}"
    )
st.image(
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
    use_container_width=True
)
st.markdown(
"""
### Features Used
✔ City

✔ Average Cost For Two

✔ Price Range

✔ Votes

✔ Aggregate Rating

✔ Table Booking

✔ Online Delivery
"""
)