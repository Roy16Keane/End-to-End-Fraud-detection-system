import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000/predict")

#API_URL = "http://127.0.0.1:8000/predict"

st.title("Fraud Detection Demo")

TransactionDT = st.number_input("TransactionDT", value=100000)
TransactionAmt = st.number_input("Transaction Amount", value=50.0)
ProductCD = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"])
card1 = st.number_input("card1", value=1234)
addr1 = st.number_input("addr1", value=200)
email = st.text_input("Email domain", "gmail.com")

if st.button("Predict"):
    payload = {
        "transaction": {
            "TransactionDT": TransactionDT,
            "TransactionAmt": TransactionAmt,
            "ProductCD": ProductCD,
            "card1": card1,
            "addr1": addr1,
            "P_emaildomain": email,
        }
    }

    r = requests.post(API_URL, json=payload)

    if r.status_code == 200:
        st.success(r.json())
    else:
        st.error(r.text)
