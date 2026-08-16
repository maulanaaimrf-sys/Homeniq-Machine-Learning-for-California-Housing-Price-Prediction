import pickle
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# ------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("XGBoostModel_CaliforniaHouse.sav", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🏡 Machine Learning for California Housing Price Prediction")
st.markdown("##### Machine Learning to Predict Property Prices in California")
st.write("Enter the following details to estimate a property's price:")

st.divider()

# ------------------------------------------------------------------
# INPUT LAYOUT (2 columns, like the reference design)
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    median_income = st.number_input(
        "Median Income (in tens of thousands USD)",
        min_value=0.5, max_value=15.0, value=3.80, step=0.01,
        help="Min: 0.5 | Max: 15.0"
    )

    housing_median_age = st.number_input(
        "Housing Median Age (years)",
        min_value=1, max_value=52, value=29, step=1,
        help="Min: 1 | Max: 52"
    )

    total_rooms = st.number_input(
        "Total Rooms in the Block",
        min_value=2, max_value=32627, value=2640, step=1,
        help="Min: 2 | Max: 32627"
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms in the Block",
        min_value=1, max_value=6445, value=538, step=1,
        help="Min: 1 | Max: 6445"
    )

    population = st.number_input(
        "Population in the Block",
        min_value=3, max_value=35682, value=1425, step=1,
        help="Min: 3 | Max: 35682"
    )

with col2:
    households = st.number_input(
        "Number of Households in the Block",
        min_value=1, max_value=6082, value=499, step=1,
        help="Min: 1 | Max: 6082"
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.54, max_value=41.95, value=35.63, step=0.01, format="%.2f",
        help="Min: 32.54 | Max: 41.95"
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-124.35, max_value=-114.31, value=-119.57, step=0.01, format="%.2f",
        help="Min: -124.35 | Max: -114.31"
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        options=["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY"]
    )

st.divider()

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
if st.button("🔍 Predict Property Price", use_container_width=False):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated property price: **${prediction:,.2f}**")

st.divider()
st.caption("Created by Maulana Imam Rifai | 2026")
