import lightgbm as lgb
import pandas as pd
import streamlit as st
import numpy as np

# Load the Mortality model for stroke patients
mortality_model = lgb.Booster(model_file='stroke_mortality_model.txt')

# Mapping for Urine_protein and AKIGrade values
ACEI_ARB_mapping = {"No": 0, "Yes": 1}
AKI_and_AKD_mapping = {"NKD": 0, "AKI recovery": 1, "subacute AKD": 2, "AKD with AKI": 3}
Diuretic_mapping = {"No": 0, "Yes": 1}
Antibiotic_mapping = {"No": 0, "Yes": 1}

def predict_mortality_probability(features):
    mortality_prob = mortality_model.predict(features)
    return mortality_prob[0]

def main():
    st.title('Mortality Probability Prediction after Stroke')

    # User selects which content to display
    selected_content = st.radio("", ("Model Introduction", "Prediction"))

    if selected_content == "Model Introduction":
        st.subheader("Model Introduction")
        st.write("This online platform provides prediction for the probability of mortality after stroke using a LightGBMmodel.")
        # Disclaimer
        st.subheader("Disclaimer")
        st.write("The predictions generated by this model are based on historical data and statistical patterns, and they may not be entirely accurate or applicable to every individual.")
        st.write("**For Patients:**")
        st.write("- The predictions presented by this platform are intended for informational purposes only and should not be regarded as a substitute for professional medical advice, diagnosis, or treatment.")
        st.write("- Consult with your healthcare provider for personalized medical guidance and decisions concerning your health.")
        st.write("**For Healthcare Professionals:**")
        st.write("- This platform should be considered as a supplementary tool to aid clinical decision-making and should not be the sole determinant of patient care.")
        st.write("- Clinical judgment and expertise should always take precedence in medical practice.")
        st.write("**For Researchers:**")
        st.write("- While this platform can serve as a valuable resource for research purposes, it is crucial to validate its predictions within your specific clinical context and patient population.")
        st.write("- Ensure that your research adheres to all ethical and regulatory standards.")
        st.write("The creators of this online platform and model disclaim any responsibility for decisions or actions taken based on the predictions provided herein. Please use this tool responsibly and always consider individual patient characteristics and clinical context when making medical decisions.")
        st.write("By utilizing this online platform, you agree to the terms and conditions outlined in this disclaimer.")

    elif selected_content == "Prediction":
        st.subheader("Mortality Prediction after Stroke")

        # Feature input
        features = []

        st.subheader("Features")

        ACEI_ARB = st.selectbox("ACEI/ARB", ["No", "Yes"]) 
        AKI_and_AKD = st.selectbox("Renal function trajectories", ["NKD", "AKI recovery", "subacute AKD", "AKD with AKI"])
        NEUT=st.number_input("Neutrophil count (×10^9/L)", value=0.00, format="%.2f")
        Diuretic=st.selectbox("Diuretics", ["No", "Yes"]) 
        Scr=st.number_input("Scr (μmol/L)", value=0.00, format="%2f")
        Antibiotic=st.selectbox("Antibiotics", ["No", "Yes"]) 
        Lipoprotein_a=st.number_input("Lipoprotein_a (mg/L)", value=0, format="%d")
        Na=st.number_input("Sodium (mmol/L)", value=0, format="%d")
        K=st.number_input("Potassium (mmol/L)", value=0.00, format="%2f")
        Mg=st.number_input("Magnesium (mmol/L)", value=0.00, format="%2f")

       # Map
        ACEI_ARB_encoded = ACEI_ARB_mapping[ACEI_ARB]
        AKI_and_AKD_encoded = AKI_and_AKD_mapping[AKI_and_AKD]
        Diuretic_encoded = Diuretic_mapping[Diuretic]
        Antibiotic_encoded = Antibiotic_mapping[Antibiotic]
        features.extend([ACEI_ARB_encoded, AKI_and_AKD_encoded, NEUT, Diuretic_encoded, Scr, Antibiotic_encoded, Lipoprotein_a, Na, K, Mg])

        # Create a button to make predictions
        if st.button('Predict Mortality'):
            features_array = np.array(features).reshape(1, -1)
            mortality_probability = predict_mortality_probability(features_array)
            st.write(f'Mortality Probability for Stroke Patients: {mortality_probability:.2f}')

if __name__ == '__main__':
    main()
