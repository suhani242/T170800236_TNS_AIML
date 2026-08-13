import streamlit as st
import pandas as pd
import joblib


# ==========================================
# 1. LOAD MODEL AND PREPROCESSING OBJECTS
# ==========================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Manufacturing Output Prediction",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Manufacturing Output Prediction")

st.write(
    "Enter the manufacturing parameters to predict Parts Per Hour."
)


# ==========================================
# 3. NUMERICAL INPUTS
# ==========================================

st.header("Machine Parameters")

injection_temperature = st.number_input(
    "Injection Temperature",
    min_value=0.0,
    value=220.0
)

injection_pressure = st.number_input(
    "Injection Pressure",
    min_value=0.0,
    value=80.0
)

cycle_time = st.number_input(
    "Cycle Time",
    min_value=0.0,
    value=30.0
)

cooling_time = st.number_input(
    "Cooling Time",
    min_value=0.0,
    value=10.0
)

material_viscosity = st.number_input(
    "Material Viscosity",
    min_value=0.0,
    value=120.0
)

ambient_temperature = st.number_input(
    "Ambient Temperature",
    value=25.0
)

machine_age = st.number_input(
    "Machine Age",
    min_value=0.0,
    value=5.0
)

operator_experience = st.number_input(
    "Operator Experience",
    min_value=0.0,
    value=3.0
)

maintenance_hours = st.number_input(
    "Maintenance Hours",
    min_value=0.0,
    value=100.0
)

temperature_pressure_ratio = st.number_input(
    "Temperature Pressure Ratio",
    min_value=0.0,
    value=2.75
)

total_cycle_time = st.number_input(
    "Total Cycle Time",
    min_value=0.0,
    value=40.0
)

efficiency_score = st.number_input(
    "Efficiency Score",
    min_value=0.0,
    value=80.0
)

machine_utilization = st.number_input(
    "Machine Utilization",
    min_value=0.0,
    value=85.0
)


# ==========================================
# 4. CATEGORICAL INPUTS
# ==========================================

st.header("Categorical Parameters")

shift = st.selectbox(
    "Shift",
    ["Day", "Evening", "Night"]
)

machine_type = st.selectbox(
    "Machine Type",
    ["Type_A", "Type_B", "Type_C"]
)

material_grade = st.selectbox(
    "Material Grade",
    ["Economy", "Premium", "Standard"]
)

day_of_week = st.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)


# ==========================================
# 5. PREDICTION
# ==========================================

if st.button("Predict Parts Per Hour"):

    # -------------------------------
    # Create input dataframe
    # -------------------------------

    input_data = pd.DataFrame({
        "Injection_Temperature": [injection_temperature],
        "Injection_Pressure": [injection_pressure],
        "Cycle_Time": [cycle_time],
        "Cooling_Time": [cooling_time],
        "Material_Viscosity": [material_viscosity],
        "Ambient_Temperature": [ambient_temperature],
        "Machine_Age": [machine_age],
        "Operator_Experience": [operator_experience],
        "Maintenance_Hours": [maintenance_hours],

        "Shift": [shift],
        "Machine_Type": [machine_type],
        "Material_Grade": [material_grade],
        "Day_of_Week": [day_of_week],

        "Temperature_Pressure_Ratio": [
            temperature_pressure_ratio
        ],

        "Total_Cycle_Time": [total_cycle_time],

        "Efficiency_Score": [
            efficiency_score
        ],

        "Machine_Utilization": [
            machine_utilization
        ]
    })


    # ==========================================
    # 6. LABEL ENCODING
    # ==========================================

    categorical_columns = [
        "Shift",
        "Machine_Type",
        "Material_Grade",
        "Day_of_Week"
    ]

    for column in categorical_columns:

        input_data[column] = label_encoders[column].transform(
            input_data[column]
        )


    # ==========================================
    # 7. FEATURE SCALING
    # ==========================================

    numerical_columns = [
        "Injection_Temperature",
        "Injection_Pressure",
        "Cycle_Time",
        "Cooling_Time",
        "Material_Viscosity",
        "Ambient_Temperature",
        "Machine_Age",
        "Operator_Experience",
        "Maintenance_Hours",
        "Temperature_Pressure_Ratio",
        "Total_Cycle_Time",
        "Efficiency_Score",
        "Machine_Utilization"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )


    # ==========================================
    # 8. PREDICTION
    # ==========================================

    prediction = model.predict(input_data)


    # ==========================================
    # 9. DISPLAY RESULT
    # ==========================================

    st.success(
        f"Predicted Parts Per Hour: {prediction[0]:.2f}"
    )