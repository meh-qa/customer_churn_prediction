import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman : ', ('EDA', 'Churn Predict'))

if navigation == 'EDA':
    eda.run()
elif navigation == 'Churn Predict':
    prediction.run()