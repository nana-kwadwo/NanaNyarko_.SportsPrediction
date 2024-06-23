import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.joblib')  
features_list = ['potential', 'value_eur', 'wage_eur', 'age', 'release_clause_eur',
                 'shooting', 'passing', 'dribbling', 'physic', 'attacking_short_passing',
                 'skill_curve', 'skill_long_passing', 'skill_ball_control',
                 'movement_reactions', 'power_shot_power', 'power_long_shots',
                 'mentality_vision', 'mentality_composure']


def predict(features):
    df = pd.DataFrame([features], columns=features.keys())
    prediction = model.predict(df)
    return prediction[0]  

# Streamlit app
st.title("FIFA Rating Predictor")

st.write("Enter the values for the 18 numerical features:")

# Create input fields for each feature
features = {}
for feature_name in features_list:
    features[feature_name] = st.number_input(feature_name, value=0.0)

# Button for making prediction
if st.button("Predict"):
    prediction = predict(features)
    st.write(f"Prediction: {prediction}")
