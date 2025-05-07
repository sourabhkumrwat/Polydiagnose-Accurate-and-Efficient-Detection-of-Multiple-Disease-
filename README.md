Predict whether a person is likely to have a specific disease based on clinical parameters using a trained Random Forest model.
Based on a publicly available dataset containing medical attributes such as blood pressure, specific gravity, hemoglobin, and serum creatinine.
A RandomForestClassifier was trained with predict_proba support. Achieved high classification accuracy after preprocessing and cleaning.
The final model is exported as a .sav file and used in a Streamlit web app that gives both prediction and probability of disease presence.
