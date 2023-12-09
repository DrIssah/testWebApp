# app.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Load the data
file_name = "social_media_usage.csv"
s = pd.read_csv(file_name)

# Define the cleaning methods
def clean_sm(x):
    for column in x.columns:
        df_arr = x[column]
        x[column] = np.where(df_arr == 1, 1, 0)
    return x

def check_greater_than(column, max_value):
    condition = (column >= 1) & (column <= max_value)
    clean_column = np.where(condition, column, np.nan)
    return clean_column

def check_greater_than_v2(column, value):
    condition = column == value
    clean_column = np.where(condition, 1, 0)
    return clean_column

# Apply cleaning methods
ss = pd.DataFrame({"sm_li": s['web1h']})
ss = clean_sm(ss)

ss['income'] = check_greater_than(s['income'],9)
ss['education'] = check_greater_than(s['educ2'],8)

##parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing)
ss['parent'] = check_greater_than_v2(s['par'],1)
ss['married'] = check_greater_than_v2(s['marital'],1)
ss['female'] = check_greater_than_v2(s['gender'],2)

ss['age'] = check_greater_than(s['age'],98)

#drop nulls
ss = ss.dropna()

# Mapping for coded values
income_mapping = {
    1: "Less than $10,000",
    2: "$10,000 to $20,000",
    3: "$20,000 to $30,000",
    4: "$30,000 to $40,000",
    5: "$40,000 to $50,000",
    6: "$50,000 to $75,000",
    7: "$75,000 to $100,000",
    8: "$100,000 to $150,000",
    9: "$150,000 or more",
    98: "(VOL.) Don't know",
    99: "(VOL.) Refused"
}

education_mapping = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college or university degree",
    7: "Some postgraduate or professional schooling",
    8: "Postgraduate or professional degree",
    98: "Don’t know",
    99: "Refused"
}

parent_mapping = {
    1: "Yes",
    2: "No"
}

marital_mapping = {
    1: "Married",
    2: "No"
    
}

gender_mapping = {
    1: "Female",
    2: "Not Female"
}



# Prepare data for modeling
y = ss['sm_li']
X = ss.drop(['sm_li'], axis=1)

# Instantiate logistic regression model with balanced class weights
model = LogisticRegression(class_weight='balanced')

# Fit the model with all data
model.fit(X, y)

# Streamlit app
st.title("LinkedIn User Prediction App")

# Sidebar with input fields
st.sidebar.header("User Input Features")

# Define input fields
income = st.sidebar.slider("Income", 1, 9, 5)
education = st.sidebar.slider("Education", 1, 8, 4)
parent = st.sidebar.radio("Parent[1.Parent 2. Not Parent]", [1, 2])
married = st.sidebar.radio("Marital Status [1. Married 2. Not Married]", [1, 2])
female = st.sidebar.radio("Female [1. Female 2. Not Female]", [1, 2])
age = st.sidebar.slider("Age", 18, 98, 42)

# Create a DataFrame for the user input
user_input = pd.DataFrame([[income, education, parent, married, female, age]], columns=X.columns)

# Make prediction
prediction_proba = model.predict_proba(user_input)
probability_of_linkedin = prediction_proba[0][1]

# Display result
predicted_class = 1 if prediction_proba[0][1] >= 0.5 else 0
st.subheader("Prediction Result")
st.write(f"The person would be classified as a LinkedIn user: {predicted_class}")
st.write(f"Probability of being a LinkedIn user: {probability_of_linkedin:.2%}")


# Display user-friendly values in the sidebar
st.sidebar.header("User Input Features (User-Friendly)")
st.sidebar.write(f"Income: {income_mapping[income]}")
st.sidebar.write(f"Education: {education_mapping[education]}")
st.sidebar.write(f"Parent: {parent_mapping[parent]}")
st.sidebar.write(f"Marital Status: {marital_mapping[married]}")
st.sidebar.write(f"Gender: {gender_mapping[female]}")
st.sidebar.write(f"Age: {age}")
