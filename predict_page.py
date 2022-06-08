# Loading the libraries
import  streamlit as st
import pickle
import numpy as np

# loading the pickled file 
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Streamlit application
def show_predict_page():
    st.title("Software Devloper Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    # we need 2 boxes countries and education

    countries = (
        "United States of America",
        "India",
        "Germany",
        "Canada",
        "United Kingdom of Great Britain and Northern Ireland",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
        "Turkey",
        "Switzerland",
        "Israel",
        "Norway",
    )
    
    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    # creating selectbox for country and education
    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level",education)

    experience = st.slider("Years of experience", 0,50,2)

    # adding a button to start the prediction
    ok = st.button("Calculate Salary")
    if ok:
        # country, edlevel, yearscode
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        
        # Salary Prediction using pickled model
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is $ {salary[0]:.2f}")