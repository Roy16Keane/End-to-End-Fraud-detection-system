import os
from typing import Any, Dict, List

import requests
import streamlit as st
import plotly.graph_objects as go


API_URL = os.getenv(
    "API_URL",
    "http://api:8000/predict",
)

REQUEST_TIMEOUT_SECONDS = 30


st.set_page_config(
    page_title="Fraud Risk Analysis Tool demo ",
    layout="wide",
)


def format_risk_level(risk_level: str) -> str:
    risk_level = risk_level.lower()

    labels = {
        "low": "Low Risk",
        "medium": "Medium Risk",
        "high": "High Risk",
    }

    return labels.get(
        risk_level,
        risk_level.title(),
    )


def display_risk_message(
    risk_level: str,
    fraud_label: int,
    threshold: float,
) -> None:
    risk_level = risk_level.lower()

    if fraud_label == 1:
        st.error(
            "This transaction exceeded the configured fraud threshold "
            f"of {threshold:.0%} and was classified as potentially fraudulent."
        )
        return

    if risk_level == "low":
        st.success(
            "This transaction appears to have a low fraud risk and remains "
            "below the configured fraud-classification threshold."
        )
    elif risk_level == "medium":
        st.warning(
            "This transaction has a moderate fraud-risk score. Although it "
            "was not classified as fraudulent, additional review may be useful."
        )
    else:
        st.warning(
            "This transaction has a high fraud-risk score but remains below "
            "the configured classification threshold."
        )


def render_factor_card(
    factor: Dict[str, Any],
    factor_type: str,
) -> None:
    display_name = factor.get(
        "display_name",
        factor.get("feature", "Unknown feature"),
    )

    feature_group = factor.get(
        "feature_group",
        "Model signal",
    )

    business_description = factor.get(
        "business_description",
        "",
    )

    display_value = factor.get(
        "display_value",
        "Not available",
    )

    description = factor.get(
        "description",
        "",
    )

    shap_value = float(
        factor.get("shap_value", 0.0)
    )

    impact_label = (
        "Risk-increasing impact"
        if factor_type == "risk"
        else "Risk-reducing impact"
    )

    with st.container(border=True):
        st.markdown(f"#### {display_name}")

        col1, col2, col3 = st.columns(
            [1.4, 1, 1]
        )

        with col1:
            st.caption("Feature group")
            st.write(feature_group)

        with col2:
            st.caption("Transaction value")
            st.write(display_value)

        with col3:
            st.caption(impact_label)
            st.write(f"{shap_value:+.3f}")

        if business_description:
            st.caption(business_description)

        if description:
            st.write(description)


def render_factor_section(
    title: str,
    factors: List[Dict[str, Any]],
    factor_type: str,
) -> None:
    st.subheader(title)

    if not factors:
        st.info(
            "No material factors were returned for this category."
        )
        return

    for factor in factors:
        render_factor_card(
            factor=factor,
            factor_type=factor_type,
        )


def render_technical_details(
    explanation: Dict[str, Any],
) -> None:
    with st.expander("Technical explanation details"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Explanation method",
                explanation.get(
                    "method",
                    "Not available",
                ),
            )

        with col2:
            base_value = explanation.get(
                "base_value"
            )

            if base_value is not None:
                st.metric(
                    "Model baseline",
                    f"{float(base_value):.3f}",
                )
            else:
                st.metric(
                    "Model baseline",
                    "Not available",
                )

        with col3:
            st.metric(
                "Output space",
                explanation.get(
                    "output_space",
                    "Not available",
                ),
            )

        st.info(
            "The baseline and feature contributions are expressed in the "
            "model's raw score space rather than as direct probability "
            "percentage changes."
        )

        st.markdown("#### Risk-increasing contributions")

        risk_rows = []

        for factor in explanation.get(
            "top_risk_factors",
            [],
        ):
            risk_rows.append(
                {
                    "Feature": factor.get(
                        "feature"
                    ),
                    "Display name": factor.get(
                        "display_name"
                    ),
                    "Value": factor.get(
                        "display_value"
                    ),
                    "SHAP contribution": factor.get(
                        "shap_value"
                    ),
                }
            )

        if risk_rows:
            st.dataframe(
                risk_rows,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Risk-reducing contributions")

        protective_rows = []

        for factor in explanation.get(
            "top_protective_factors",
            [],
        ):
            protective_rows.append(
                {
                    "Feature": factor.get(
                        "feature"
                    ),
                    "Display name": factor.get(
                        "display_name"
                    ),
                    "Value": factor.get(
                        "display_value"
                    ),
                    "SHAP contribution": factor.get(
                        "shap_value"
                    ),
                }
            )

        if protective_rows:
            st.dataframe(
                protective_rows,
                use_container_width=True,
                hide_index=True,
            )
def render_executive_summary(
    explanation: Dict[str, Any],
    ) -> None:

    summary = explanation.get(
        "executive_summary",
        {},
    )

    if not summary:
        return

    st.subheader("Fraud Analyst Summary")

    headline = summary.get(
        "headline",
        "Transaction Assessment",
    )

    st.markdown(f"### {headline}")

    assessment = summary.get("assessment")

    if assessment:
        st.write(assessment)

    key_drivers = summary.get("key_drivers")

    if key_drivers:
        st.markdown("**Key model drivers**")
        st.write(key_drivers)

    data_quality = summary.get("data_quality")

    if data_quality:
        st.markdown("**Data quality observation**")
        st.write(data_quality)

    suggested_action = summary.get(
        "suggested_action"
    )

    if suggested_action:
        st.markdown("**Suggested analyst action**")
        st.info(suggested_action)

def render_shap_waterfall(
    explanation: Dict[str, Any],
   ) -> None:
    """
    Render a local TreeSHAP waterfall showing how the model moved
    from its baseline raw score to the final transaction score.
    """

    waterfall = explanation.get("waterfall", {})

    if not waterfall:
        st.info(
            "Waterfall explanation data is not available."
        )
        return

    base_value = float(
        waterfall.get("base_value", 0.0)
    )

    final_raw_score = float(
        waterfall.get("final_raw_score", 0.0)
    )

    other_contribution = float(
        waterfall.get(
            "other_features_contribution",
            0.0,
        )
    )

    factors = waterfall.get("factors", [])

    labels = ["Model baseline"]
    values = [base_value]
    measures = ["absolute"]

    for factor in factors:
        display_name = factor.get(
            "display_name",
            factor.get("feature", "Model feature"),
        )

        shap_value = float(
            factor.get("shap_value", 0.0)
        )

        labels.append(display_name)
        values.append(shap_value)
        measures.append("relative")

    if abs(other_contribution) > 1e-10:
        labels.append("Other model signals")
        values.append(other_contribution)
        measures.append("relative")

    labels.append("Final model score")
    values.append(0)
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={
                "line": {
                    "width": 1,
                }
            },
            text=[
                f"{value:+.3f}"
                if index > 0
                else f"{value:.3f}"
                for index, value in enumerate(values)
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="How the Model Reached This Assessment",
        xaxis_title="Model signals",
        yaxis_title="Contribution to raw model score",
        showlegend=False,
        height=550,
        margin=dict(
            l=40,
            r=40,
            t=80,
            b=160,
        ),
    )

    fig.update_xaxes(
        tickangle=-35
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        f"The model started from a baseline raw score of "
        f"{base_value:.3f} and ended at {final_raw_score:.3f}. "
        "Positive contributions move the assessment toward greater "
        "fraud risk, while negative contributions move it toward "
        "lower fraud risk."
    )


st.title("Fraud Risk Analysis demo")

st.markdown(
    """
    This application uses a machine learning model to estimate the likelihood
    that a transaction may be fraudulent.

    Enter the transaction information below, select a classification threshold,
    and review both the prediction and the factors that influenced it.
    """
)

st.info(
    "Try changing the transaction amount, card details, address values, "
    "or email domain to observe how the fraud score and explanation change."
)

st.divider()

st.subheader("Transaction details")

st.caption(
    "The values below represent transaction information used by the machine learning model. "
    "Several fields are intentionally anonymised because the original IEEE-CIS fraud dataset masks sensitive customer information."
)

with st.expander("What do these inputs mean?"):
    st.markdown(
        """
        ### Transaction time
        Represents when the transaction occurred within the original dataset.

        It is **not a real date or timestamp**, but a numerical value used to preserve the order and timing of transactions.

        ---

        ### Transaction amount
        The monetary value of the transaction.

        The model learns spending patterns from historical data, making unusually small or large amounts useful fraud indicators.

        ---

        ### Product category
        An anonymised code representing the type of product or service involved in the transaction.

        Different product categories may exhibit different fraud patterns.

        ---

        ### Primary card attribute
        An anonymised identifier associated with the customer's payment card.

        Although its real meaning is hidden for privacy, it helps the model recognise historical card behaviour.

        ---

        ### Secondary card attribute
        An additional anonymised characteristic associated with the payment card.

        Together with other card-related attributes, it helps identify normal and unusual transaction patterns.

        ---

        ### Card country attribute
        An anonymised geographical attribute linked to the payment card.

        The model compares this with historical transactions when assessing fraud risk.

        ---

        ### Card issuer attribute
        An anonymised attribute representing characteristics of the organisation that issued the payment card.

        Historical issuer behaviour provides additional context for the model.

        ---

        ### Billing-region attribute
        An anonymised billing-region identifier associated with the transaction.

        The model uses this to compare the transaction with previous activity from similar regions.

        ---

        ### Billing-country attribute
        An anonymised billing-country identifier.

        Combined with other transaction details, it helps identify transactions that differ from expected customer behaviour.

        ---

        ### Purchaser email domain
        The email provider used by the person making the purchase.

        Examples include:

        - gmail.com
        - outlook.com
        - yahoo.com

        The model has learned historical patterns associated with different email domains.

        ---

        ### Recipient email domain
        The email provider associated with the recipient of the transaction.

        In many purchases this will be the same as the purchaser's email address. However, it may differ when someone is buying an item or gift for another person.

        The relationship between purchaser and recipient information can provide useful context when assessing fraud risk.
        """
    )


input_col1, input_col2 = st.columns(2)

with input_col1:
    transaction_dt = st.number_input(
        "Transaction time",
        min_value=0,
        value=100000,
        step=1000,
    )

    transaction_amt = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=150.0,
        step=10.0,
        format="%.2f",
    )

    product_cd = st.selectbox(
        "Product category",
        ["W", "C", "R", "H", "S"],
    )

    card1 = st.number_input(
        "Primary card attribute",
        min_value=0,
        value=1001,
        step=1,
    )

    card2 = st.number_input(
        "Secondary card attribute",
        min_value=0,
        value=321,
        step=1,
    )

with input_col2:
    card3 = st.number_input(
        "Card country attribute",
        min_value=0,
        value=150,
        step=1,
    )

    card5 = st.number_input(
        "Card issuer attribute",
        min_value=0,
        value=226,
        step=1,
    )

    addr1 = st.number_input(
        "Billing-region attribute",
        min_value=0,
        value=315,
        step=1,
    )

    addr2 = st.number_input(
        "Billing-country attribute",
        min_value=0,
        value=87,
        step=1,
    )

    purchaser_email = st.text_input(
        "Purchaser email domain",
        value="gmail.com",
    )

    recipient_email = st.text_input(
        "Recipient email domain",
        value="gmail.com",
    )

st.divider()

settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    threshold = st.slider(
        "Fraud classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Transactions with a fraud probability equal to or above this "
            "value will be classified as potentially fraudulent."
        ),
    )

with settings_col2:
    max_explanation_features = st.slider(
        "Number of explanation factors",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

st.caption(
    f"A transaction will be classified as potentially fraudulent when its "
    f"fraud probability reaches {threshold:.0%} or higher."
)

predict_button = st.button(
    "Analyse Transaction",
    type="primary",
    use_container_width=True,
)

if predict_button:
    payload = {
        "transaction": {
            "TransactionDT": transaction_dt,
            "TransactionAmt": transaction_amt,
            "ProductCD": product_cd,
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card5": card5,
            "addr1": addr1,
            "addr2": addr2,
            "P_emaildomain": purchaser_email,
            "R_emaildomain": recipient_email,
        },
        "threshold": threshold,
        "explain": True,
        "max_explanation_features": (
            max_explanation_features
        ),
    }

    try:
        with st.spinner(
            "Analysing the transaction..."
        ):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout:
        st.error(
            "The prediction API took too long to respond."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "The application could not connect to the prediction API. "
            "Check that the API container is running and healthy."
        )

    except requests.exceptions.HTTPError:
        st.error(
            f"The prediction API returned status "
            f"{response.status_code}."
        )
        st.code(response.text)

    except requests.exceptions.RequestException as error:
        st.error(
            "An unexpected API connection error occurred."
        )
        st.code(str(error))

    except ValueError:
        st.error(
            "The API returned a response that could not be decoded."
        )

    else:
        fraud_probability = float(
            result.get("fraud_proba", 0.0)
        )

        fraud_label = int(
            result.get("fraud_label", 0)
        )

        returned_threshold = float(
            result.get(
                "threshold",
                threshold,
            )
        )

        risk_level = result.get(
            "risk_level",
            "unknown",
        )

        explanation = result.get(
            "explanation",
            {},
        )

        st.divider()
        st.header("Assessment result")

        metric_col1, metric_col2, metric_col3 = (
            st.columns(3)
        )

        with metric_col1:
            st.metric(
                "Fraud probability",
                f"{fraud_probability:.1%}",
            )

        with metric_col2:
            st.metric(
                "Risk level",
                format_risk_level(risk_level),
            )

        with metric_col3:
            st.metric(
                "Model classification",
                (
                    "Potential fraud"
                    if fraud_label == 1
                    else "Not classified as fraud"
                ),
            )

        st.progress(
            min(
                max(fraud_probability, 0.0),
                1.0,
            )
        )

        st.caption(
            f"Configured fraud-classification threshold: "
            f"{returned_threshold:.0%}"
        )

        display_risk_message(
            risk_level=risk_level,
            fraud_label=fraud_label,
            threshold=returned_threshold,
        )

        render_executive_summary(explanation)

        st.divider()
        st.header("Why the model made this assessment")
        st.markdown(
            """
            The chart below shows how the most influential model signals
            moved this transaction away from the model's typical baseline
            assessment.
            """
        )

        render_shap_waterfall(explanation)

        st.divider()

        risk_tab, protective_tab = st.tabs(
            [
                "Risk-increasing factors",
                "Risk-reducing factors",
            ]
        )

        with risk_tab:
            render_factor_section(
                title="Signals increasing the risk score",
                factors=explanation.get(
                    "top_risk_factors",
                    [],
                ),
                factor_type="risk",
            )

        with protective_tab:
            render_factor_section(
                title="Signals reducing the risk score",
                factors=explanation.get(
                    "top_protective_factors",
                    [],
                ),
                factor_type="protective",
            )

        disclaimer = explanation.get("disclaimer")

        if disclaimer:
            st.info(disclaimer)

        render_technical_details(explanation)

        with st.expander("Raw API response"):
            st.json(result)