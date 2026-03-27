import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load(r"D:\medical_cost_prediction\Main Folder\models\final_model.pkl")

feature_columns = joblib.load(r"D:\medical_cost_prediction\Main Folder\models\feature_columns.pkl")

st.title("💊 Medical Cost Prediction App")
st.write("Predict insurance charges based on user details")


age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])


if st.button("Predict Medical Cost"):

    # Create DataFrame
    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker]
    })

    new_data["age_bmi_interaction"] = new_data["age"] * new_data["bmi"]
    new_data["sex"] = new_data["sex"].map({"female": 0, "male": 1})
    new_data["smoker"] = new_data["smoker"].map({"no": 0, "yes": 1})

    if new_data[["sex", "smoker"]].isnull().values.any():
        st.error("Invalid category detected!")
    else:
      
        new_data = new_data.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(new_data)

        st.success(f"💰 Predicted Medical Cost: ${prediction[0]:,.2f}")