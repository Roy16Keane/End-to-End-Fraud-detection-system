import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000/predict")

st.set_page_config(
    page_title="Fraud Detection Demo",
    layout="centered"
)

st.title("Fraud Detection Demo")

st.markdown("""
This demo uses a machine learning model to estimate whether a transaction is likely to be fraudulent.

### How to use the demo
1. Enter the transaction details below.
2. Click **Predict Fraud Risk**.
3. Review the fraud probability, risk level, and model prediction.

### How to interpret the result
- **Fraud probability** shows how risky the transaction looks to the model.
- A value closer to **0%** means lower fraud risk.
- A value closer to **100%** means higher fraud risk.
- The final label shows whether the model classifies the transaction as fraudulent or not.
""")

st.info("Tip: Try changing the transaction amount, card value, address value, or email domain to see how the model response changes.")

st.divider()

st.subheader("Enter Transaction Details")

with st.expander("ℹ️ What do these inputs mean?"):
    st.markdown("""
    - **TransactionDT**  
      Represents when the transaction happened in the dataset.  
      It is not a normal date or clock time, but a numerical time value used to track transaction order.

    - **Transaction Amount**  
      The amount of money being spent in the transaction.

    - **ProductCD**  
      A product category code showing the type of product or service involved in the transaction.

    - **card1**  
      An encoded card-related value.  
      It helps the model detect patterns linked to card usage.

    - **addr1**  
      An encoded address-related value.  
      It helps the model identify location or billing-address patterns.

    - **Email domain**  
      The customer’s email provider, such as `gmail.com`, `yahoo.com`, or `hotmail.com`.
    """)

TransactionDT = st.number_input("TransactionDT", value=100000)
TransactionAmt = st.number_input("Transaction Amount", value=50.0)
ProductCD = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"])
card1 = st.number_input("card1", value=1234)
addr1 = st.number_input("addr1", value=200)
email = st.text_input("Email domain", "gmail.com")

st.divider()

def get_risk_level(probability):
    if probability < 0.30:
        return "Low Risk", "green"
    elif probability < 0.70:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

if st.button("Predict Fraud Risk"):
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

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            fraud_proba = result.get("fraud_proba")
            fraud_label = result.get("fraud_label")
            threshold = result.get("threshold", 0.5)

            st.subheader("Prediction Result")

            if fraud_proba is not None:
                risk_level, risk_colour = get_risk_level(fraud_proba)

                st.metric("Fraud Probability", f"{fraud_proba:.2%}")

                st.progress(float(fraud_proba))

                st.markdown(f"""
                ### Risk Level: :{risk_colour}[{risk_level}]
                """)

                if fraud_proba < 0.30:
                    st.success("This transaction appears to have a low fraud risk.")
                elif fraud_proba < 0.70:
                    st.warning("This transaction has a moderate fraud risk. It may require further review.")
                else:
                    st.error("This transaction appears to have a high fraud risk and should be investigated carefully.")

            if fraud_label is not None:
                if fraud_label == 1:
                    st.error("Model Decision: Fraudulent transaction predicted.")
                else:
                    st.success("Model Decision: Transaction predicted as not fraudulent.")

            st.caption(f"Decision threshold used by the model: {threshold}")

            with st.expander("View raw API response"):
                st.json(result)

        else:
            st.error("The API returned an error.")
            st.code(response.text)

    except requests.exceptions.RequestException as e:
        st.error("Could not connect to the prediction API.")
        st.code(str(e))
