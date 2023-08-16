import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load Model
with open('final_pipeline.pkl', 'rb') as file_1:
  final_pipeline = pickle.load(file_1)

model_ann = load_model('churn_model.h5')
  
# with open('best_model_logreg.pkl', 'rb') as file_2:
#   logreg_best_model = pickle.load(file_2)


def run():
    # Create Form Input
    with st.form('key=form_sleep_disorder'):
        age = st.number_input('Age', min_value=0, max_value=200, value=25)
        gender = st.radio('Gender', ('M', 'F'))
        membership_category = st.radio('Membership', ('No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'))
        preferred_offer_types = st.radio('Prefered Offer Type', ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'))
        internet_option = st.radio('Internet', ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'))
        days_since_last_login = st.number_input('Days since last login', min_value=0, max_value=365, value=0)
        avg_time_spent = st.number_input('Average time spent', min_value=0, max_value=999, value=0)
        avg_transaction_value = st.number_input('Average transaction value', min_value=0, max_value=99999999999, value=0)
        avg_frequency_login_days = st.number_input('Average frequency login days', min_value=0, max_value=365, value=0)
        points_in_wallet = st.number_input('Points in wallet', min_value=0, max_value=99999999999, value=0)
        used_special_discount = st.radio('Use Special Discount?', ('Yes', 'No'))
        offer_application_preference = st.radio('Offer Apps Preference', ('Yes', 'No'))
        past_complaint = st.radio('Any Past Complaint?', ('Yes', 'No'))
        complaint_status = st.radio('Complaint Status', ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'))
        feedback = st.radio('feedback', ('Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'))

        submitted = st.form_submit_button('Predict')

    data_inf = {
      'age':age,
      'gender': gender,
      'membership_category': membership_category,
      'preferred_offer_types': preferred_offer_types,
      'internet_option': internet_option,
      'days_since_last_login': days_since_last_login,
      'avg_time_spent': avg_time_spent,
      'avg_transaction_value': avg_transaction_value,
      'avg_frequency_login_days': avg_frequency_login_days,
      'points_in_wallet': points_in_wallet,
      'used_special_discount': used_special_discount,
      'offer_application_preference': offer_application_preference,
      'past_complaint': past_complaint,
      'complaint_status': complaint_status,
      'feedback': feedback
    }


    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Predict using ANN

        data_inf_transform = final_pipeline.transform(data_inf)
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        y_pred_inf

        st.write('# Predicted Churn Score: ', str(y_pred_inf))
    
if __name__ == '__main__':
    run()