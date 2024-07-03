import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.title('Hello World')
st.write('This is an app to check if email is spam or not.')

email = st.text_area('Enter email text')
if st.button('Check'):
    if email == '':
        st.write('Please enter email text')
    else:      
        tfidf_vectorizer = joblib.load(open('tfidf_vectorizer.pkl', 'rb'))
        email_tfidf = tfidf_vectorizer.transform([email])
        model = joblib.load(open('spam_classifier_model.pkl', 'rb'))
        prediction = model.predict(email_tfidf)[0]
        st.write('Email is spam' if prediction else 'Email is not spam')

           
# Define sample data
data ={
    'year': [2019, 2020, 2021, 2022],
    'spam_count': [779200, 1845814, 2847773, 4744699]
}
# Create a DataFrame
df = pd.DataFrame(data)

# Create a Streamlit app
st.title('Spam Emails Sent Around the World (Year-wise)')

# Create a line chart
fig = px.line(df, x='year', y='spam_count', title='Number of Spam Emails Sent Around the World (Year-wise)')

# Display the chart in Streamlit
st.plotly_chart(fig)

