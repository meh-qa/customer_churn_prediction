import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title= 'Churn - EDA',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)

def run():

    # Page Title
    st.title('Churn EDA')

    # Sub Header
    st.subheader('EDA for Churn Analysis')

    # Menambahkan Text
    st.write('This page is created by Mehdi')

    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Page ini, merupakan explorasi sederhana
    Dataset yang digunakan adalah dataset Sleep Disorder
    Dataset ini berasal dari kaggle
    '''

    # Show Data Frame
    data = pd.read_csv('churn.csv')
    st.dataframe(data)

    # # Membuat Barplot
    # st.write('#### Plot Age')
    # fig = plt.figure(figsize=(15,5))
    # sns.countplot(x='age', data=data)
    # st.pyplot(fig)

    # # Membuat Histogram
    # st.write('#### Histogram of Churn')
    # fig = plt.figure(figsize=(15,5))
    # sns.histplot(data['churn_risk_score'], bins=30, kde=True)
    # st.pyplot(fig)

    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram')
    pilihan = st.radio('Choose Column: ', ('age', 'gender', 'membership_category'))
    fig = plt.figure(figsize=(15,5))
    sns.countplot(data=data, x=data[pilihan], hue='churn_risk_score')
    st.pyplot(fig)


if __name__ == '__main__':
    run()