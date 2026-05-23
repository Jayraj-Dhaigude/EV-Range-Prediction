import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EV Range Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    model   = joblib.load("random_forest_model.pkl")
    columns = joblib.load("columns.pkl")
    te      = joblib.load("target_encoder.pkl")
    return model, columns, te

model, columns, te = load_artifacts()

BRANDS = ['Abarth', 'Aiways', 'Alfa', 'Alpine', 'Audi', 'BMW', 'BYD', 'CUPRA',
          'Cadillac', 'Citroen', 'DS', 'Dacia', 'Dongfeng', 'Elaris', 'Fiat',
          'Ford', 'GWM', 'Genesis', 'Honda', 'Hongqi', 'Hyundai', 'Jaguar',
          'Jeep', 'KGM', 'Kia', 'Lancia', 'Leapmotor', 'Lexus', 'Lotus',
          'Lucid', 'Lynk&Co', 'MG', 'Maserati', 'Maxus', 'Mazda',
          'Mercedes-Benz', 'Mini', 'NIO', 'Nissan', 'Omoda', 'Opel',
          'Peugeot', 'Polestar', 'Porsche', 'Renault', 'Rolls-Royce', 'Skoda',
          'Skywell', 'Smart', 'Subaru', 'Tesla', 'Toyota', 'VinFast',
          'Volkswagen', 'Volvo', 'Voyah', 'XPENG', 'Zeekr', 'firefly']

SEGMENTS = ['A - Mini', 'B - Compact', 'C - Medium', 'D - Large',
            'E - Executive', 'F - Luxury',
            'JB - Compact', 'JC - Medium', 'JD - Large',
            'JE - Executive', 'JF - Luxury', 'N - Passenger Van']

BODY_TYPES  = ['Cabriolet', 'Hatchback', 'Liftback Sedan',
               'SUV', 'Sedan', 'Small Passenger Van', 'Station/Estate']

DRIVETRAINS = ['AWD', 'FWD', 'RWD']

st.title("EV Range Predictor")
st.markdown("Fill in the vehicle specs below to predict the estimated range in km.")
st.divider()

with st.form("prediction_form"):

    st.subheader("Vehicle Identity")
    col1, col2, col3 = st.columns(3)
    with col1:
        brand       = st.selectbox("Brand", BRANDS, index=BRANDS.index("Tesla"))
    with col2:
        body_type   = st.selectbox("Body Type", BODY_TYPES, index=BODY_TYPES.index("SUV"))
    with col3:
        drivetrain  = st.selectbox("Drivetrain", DRIVETRAINS, index=0)

    segment = st.selectbox("Segment", SEGMENTS, index=SEGMENTS.index("C - Medium"))

    st.divider()
    st.subheader("Battery & Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        battery_capacity     = st.slider("Battery Capacity (kWh)", 21.0, 118.0, 75.0, step=0.5)
        fast_charging_power  = st.slider("Fast Charging Power (kW DC)", 29, 281, 150)
    with col2:
        top_speed            = st.slider("Top Speed (km/h)", 125, 325, 200)
        acceleration         = st.slider("0–100 km/h (seconds)", 2.2, 19.1, 6.0, step=0.1)
    with col3:
        torque               = st.slider("Torque (Nm)", 113, 1350, 400)
        efficiency           = st.slider("Efficiency (Wh/km)", 109, 370, 180)

    st.divider()
    st.subheader("Dimensions & Capacity")
    col1, col2, col3 = st.columns(3)
    with col1:
        length        = st.slider("Length (mm)", 3620, 5908, 4500)
        width         = st.slider("Width (mm)",  1610, 2080, 1850)
        height        = st.slider("Height (mm)", 1329, 1986, 1600)
    with col2:
        seats         = st.slider("Seats", 2, 9, 5)
        cargo_volume  = st.slider("Cargo Volume (L)", 151, 1410, 470)
    with col3:
        towing        = st.slider("Towing Capacity (kg)", 0, 2500, 1000)
        number_cells  = st.slider("Number of Cells", 72, 7920, 400)

    submitted = st.form_submit_button("⚡ Predict Range", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        'brand'                   : brand,
        'top_speed_kmh'           : top_speed,
        'battery_capacity_kWh'    : battery_capacity,
        'number_of_cells'         : number_cells,
        'torque_nm'               : torque,
        'efficiency_wh_per_km'    : efficiency,
        'acceleration_0_100_s'    : acceleration,
        'fast_charging_power_kw_dc': fast_charging_power,
        'towing_capacity_kg'      : towing,
        'cargo_volume_l'          : cargo_volume,
        'seats'                   : seats,
        'length_mm'               : length,
        'width_mm'                : width,
        'height_mm'               : height,
        'car_body_type'           : body_type,
        'segment'                 : segment,
        'drivetrain'              : drivetrain,
    }])

    input_df['brand'] = te.transform(input_df['brand'])
    input_df = pd.get_dummies(input_df, columns=['car_body_type', 'segment', 'drivetrain'])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    predicted_range = model.predict(input_df)[0]

    st.divider()
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Range", f"{predicted_range:.0f} km")
    col2.metric("Estimated Miles", f"{predicted_range * 0.621:.0f} mi")
    col3.metric("Brand", brand)

    if predicted_range < 300:
        st.info("Short range EV — best for city driving.")
    elif predicted_range < 500:
        st.success("Mid range EV — great for daily use and road trips.")
    else:
        st.success("Long range EV — excellent for long distance travel.")