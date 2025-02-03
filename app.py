import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS for design
st.markdown(
    """
    <style>
        .container { display: flex; flex-wrap: wrap; justify-content: center; }
        .input-box { width: 22%; padding: 10px; }
        .stTextInput>label, .stSelectbox>label { text-align: center; display: block; font-weight: bold; }
        .stProgress { height: 10px; }
        .cyan-text { color: #00FFFF; font-weight: bold; text-align: center; }
        
        /* Custom styling for the submit button */
        .stButton>button {
            background-color: #00FFFF !important;
            color: black !important;
            font-weight: bold;
            display: block;
            margin: auto;
            border-radius: 8px;
            padding: 8px 16px;
            transition: background-color 0.3s ease-in-out;
        }
        
        /* Hover effect */
        .stButton>button:hover {
            background-color: #0099CC !important; /* Darker blue on hover */
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title (Centered)
st.markdown("<h1 style='text-align: center; color: #00FFFF;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter Customer Details to Predict the Likelihood of Churn</p>", unsafe_allow_html=True)

# Centered form
with st.form("churn_form", clear_on_submit=False):
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    with col2:
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
    with col3:
        age = st.text_input('Age')
    with col4:
        balance = st.text_input('Balance')

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        credit_score = st.text_input('Credit Score')
    with col6:
        estimated_salary = st.text_input('Estimated Salary')
    with col7:
        tenure = st.text_input('Tenure')
    with col8:
        num_of_products = st.text_input('Number of Products')

    col9, col10 = st.columns(2)
    with col9:
        has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    with col10:
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

    st.markdown("</div>", unsafe_allow_html=True)  # End container div

    # Cyan submit button
    submit_button = st.form_submit_button("Predict Churn")

# Process input only when the form is submitted
if submit_button:
    try:
        # Convert inputs to proper types
        age = int(age)
        balance = float(balance)
        credit_score = int(credit_score)
        estimated_salary = float(estimated_salary)
        tenure = int(tenure)
        num_of_products = int(num_of_products)
        has_cr_card = 1 if has_cr_card == 'Yes' else 0
        is_active_member = 1 if is_active_member == 'Yes' else 0

        # Validate ranges
        if not (1 <= tenure <= 120):
            st.error("Tenure must be between 1 and 120.")
        elif not (1 <= num_of_products <= 50):
            st.error("Number of Products must be between 1 and 50.")
        else:
            # Prepare input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })

            # One-hot encode 'Geography'
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

            # Combine one-hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Predict churn
            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]

            # Progress bar for probability
            st.progress(float(prediction_proba))

            # Display churn probability in cyan color
            st.markdown(
                f"""
                <p class="cyan-text">Churn Probability: <b>{prediction_proba:.2%}</b></p>
                """,
                unsafe_allow_html=True
            )

            # Display churn result
            if prediction_proba > 0.5:
                st.markdown("<div style='padding:15px; border-radius:10px; background-color:#ffcccc; text-align:center;'>"
                            "<h3 style='color:#ff1a1a;'>🔴 The customer is likely to churn.</h3></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding:15px; border-radius:10px; background-color:#ccffcc; text-align:center;'>"
                            "<h3 style='color:#009933;'>🟢 The customer is not likely to churn.</h3></div>",
                            unsafe_allow_html=True)

    except ValueError:
        st.error("Please enter valid numerical values for Age, Balance, Credit Score, Salary, Tenure, and Number of Products.")
