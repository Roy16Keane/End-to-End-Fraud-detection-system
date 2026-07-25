from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.features.preprocessing import make_numeric_matrix
from fraud.utils.io import load_joblib



FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # Direct transaction fields
    "TransactionAmt": "Transaction amount",
    "ProductCD": "Product category",
    "card1": "Primary payment-card attribute",
    "card2": "Secondary payment-card attribute",
    "card3": "Payment-card country attribute",
    "card4": "Payment-card network",
    "card5": "Payment-card issuer attribute",
    "card6": "Payment-card type",
    "addr1": "Billing-region attribute",
    "addr2": "Billing-country attribute",
    "dist1": "Transaction distance indicator",
    "dist2": "Secondary distance indicator",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",

    # Your engineered features
    "card1_FE": "Payment-card usage frequency",
    "addr1_FE": "Billing-region usage frequency",
    "P_emaildomain_FE": "Purchaser email-domain frequency",
    "card1_addr1_FE": "Card and billing-region familiarity",
    "card1_addr1_P_emaildomain_FE": (
        "Card, billing-region and email familiarity"
    ),
    "TransactionAmt_card1_mean": (
        "Typical transaction amount for this card"
    ),
    "TransactionAmt_card1_std": (
        "Transaction amount variation for this card"
    ),
    "TransactionAmt_card1|addr1_mean": (
        "Typical amount for this card and billing region"
    ),
    "TransactionAmt_card1|addr1_std": (
        "Amount variation for this card and billing region"
    ),
    "TransactionAmt_P_emaildomain_mean": (
        "Typical amount for this purchaser email domain"
    ),
    "TransactionAmt_P_emaildomain_std": (
        "Amount variation for this purchaser email domain"
    ),

    # Other derived fields
    "bin": "Transaction amount band",
}



FEATURE_BUSINESS_DESCRIPTIONS: Dict[str, str] = {
    "TransactionAmt": (
        "The monetary value of the transaction."
    ),
    "ProductCD": (
        "The anonymised product category associated with the transaction."
    ),
    "card1": (
        "An anonymised attribute associated with the primary payment card."
    ),
    "card2": (
        "A secondary anonymised attribute associated with the payment card."
    ),
    "card3": (
        "An anonymised card-related geographic attribute."
    ),
    "card4": (
        "The payment-card network or category."
    ),
    "card5": (
        "An anonymised payment-card issuer attribute."
    ),
    "card6": (
        "The payment-card type or category."
    ),
    "addr1": (
        "An anonymised billing-region attribute."
    ),
    "addr2": (
        "An anonymised billing-country attribute."
    ),
    "dist1": (
        "An anonymised distance measurement associated with the transaction."
    ),
    "dist2": (
        "A secondary anonymised distance measurement."
    ),
    "P_emaildomain": (
        "The email-domain category supplied for the purchaser."
    ),
    "R_emaildomain": (
        "The email-domain category supplied for the recipient."
    ),
    "card1_FE": (
        "How frequently this payment-card attribute appeared in "
        "the model's historical training data."
    ),
    "addr1_FE": (
        "How frequently this billing-region attribute appeared in "
        "the model's historical training data."
    ),
    "P_emaildomain_FE": (
        "How frequently this purchaser email domain appeared in "
        "the model's historical training data."
    ),
    "card1_addr1_FE": (
        "How frequently this combination of payment card and billing "
        "region appeared in historical training data."
    ),
    "card1_addr1_P_emaildomain_FE": (
        "How frequently this card, billing-region and purchaser-email "
        "combination appeared in historical training data."
    ),
    "TransactionAmt_card1_mean": (
        "The historical average transaction amount associated with "
        "this payment-card attribute."
    ),
    "TransactionAmt_card1_std": (
        "The historical variation in transaction amounts associated "
        "with this payment-card attribute."
    ),
    "TransactionAmt_card1|addr1_mean": (
        "The historical average amount for this combination of card "
        "and billing region."
    ),
    "TransactionAmt_card1|addr1_std": (
        "The historical variation in amount for this combination of "
        "card and billing region."
    ),
    "bin": (
        "A grouped transaction-amount category used by the model."
    ),
}



@dataclass
class InferenceArtifacts:
    featurizer: Any
    feature_cols: list[str]
    dropped_cols: list[str]
    meta: Dict[str, Any]


class FraudPredictor:
    """
    Loads artifacts, model and SHAP explainer once and serves predictions.

    Expected files:
      artifacts/featurizer.joblib
      artifacts/train_meta.joblib
      models/xgb_model.joblib
      models/meta.json (optional)
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        model_dir: str = "models",
        model_filename: str = "xgb_model.joblib",
        meta_filename: str = "train_meta.joblib",
        featurizer_filename: str = "featurizer.joblib",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_dir = Path(model_dir)

        self.model_path = self.model_dir / model_filename
        self.meta_path = self.artifacts_dir / meta_filename
        self.featurizer_path = self.artifacts_dir / featurizer_filename

        self._artifacts: Optional[InferenceArtifacts] = None
        self._model: Any = None

    def load(self) -> "FraudPredictor":
        """
        Load the featurizer, training metadata, model and SHAP explainer.

        All objects are loaded once when the API starts.
        """
        # 1. Feature engineering object
        featurizer = load_joblib(self.featurizer_path)

        # 2. Training metadata
        meta = load_joblib(self.meta_path)

        if not isinstance(meta, dict):
            raise RuntimeError(
                "Training metadata must be stored as a dictionary."
            )

        feature_cols = meta.get("feature_cols", [])
        dropped_cols = meta.get("dropped_cols", [])

        # 3. Trained model
        model = load_joblib(self.model_path)

        self._artifacts = InferenceArtifacts(
            featurizer=featurizer,
            feature_cols=list(feature_cols) if feature_cols else [],
            dropped_cols=list(dropped_cols) if dropped_cols else [],
            meta=meta,
        )

        self._model = model


        return self

    def _ensure_loaded(self) -> None:
        if self._artifacts is None or self._model is None:
            raise RuntimeError(
                "Predictor not loaded. Call load() at startup."
            )

    
    def _prepare_features(
        self,
        payload: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply exactly the same feature engineering and column alignment
        used during model training.
        """
        self._ensure_loaded()
        assert self._artifacts is not None

        if not isinstance(payload, dict):
            raise ValueError(
                "Transaction payload must be a dictionary."
            )

        if not payload:
            raise ValueError(
                "Transaction payload cannot be empty."
            )

        df = pd.DataFrame([payload])

        # Same feature engineering as training
        df_feat = self._artifacts.featurizer.transform(df)

        # Drop identifiers and target leakage columns
        leakage_columns = [
            "TransactionID",
            "UID",
            "isFraud",
        ]

        df_feat = df_feat.drop(
            columns=[
                column
                for column in leakage_columns
                if column in df_feat.columns
            ],
            errors="ignore",
        )

        # Convert to numeric matrix and fill missing values
        X = make_numeric_matrix(df_feat)

        # Columns removed during training because they were constant
        # or entirely missing
        X = X.drop(
            columns=self._artifacts.dropped_cols,
            errors="ignore",
        )

        # Align inference data to the exact training column order
        train_cols = self._artifacts.feature_cols

        if train_cols:
            for column in train_cols:
                if column not in X.columns:
                    X[column] = -1

            # Removes unexpected extras and fixes column ordering
            X = X[train_cols]

        if X.empty:
            raise ValueError(
                "Feature preparation produced an empty feature matrix."
            )

        return X

    def predict_proba(
        self,
        payload: Dict[str, Any],
    ) -> float:
        """
        Return the positive-class fraud probability.
        """
        self._ensure_loaded()

        X = self._prepare_features(payload)

        probability = self._model.predict_proba(X)[0, 1]

        return float(probability)

    def predict(
        self,
        payload: Dict[str, Any],
        threshold: float = 0.5,
        explain: bool = False,
        max_explanation_features: int = 6,
    ) -> Dict[str, Any]:
        """
        Return a fraud prediction and optionally a local SHAP explanation.
        """
        self._ensure_loaded()

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        if not 1 <= max_explanation_features <= 20:
            raise ValueError(
                "max_explanation_features must be between 1 and 20."
            )

        # Prepare features only once.
        # The same exact row is used for prediction and explanation.
        X = self._prepare_features(payload)

        probability = float(
            self._model.predict_proba(X)[0, 1]
        )

        label = int(probability >= threshold)

        result: Dict[str, Any] = {
            "fraud_proba": probability,
            "fraud_label": label,
            "threshold": float(threshold),
            "risk_level": self._get_risk_level(probability),
        }

        if explain:
            result["explanation"] = self.explain_prediction(
                X=X,
                predicted_probability=probability,
                threshold=threshold,
                max_features=max_explanation_features,
            )

        return result

    def _build_executive_summary(
        self,
        probability: float,
        threshold: float,
        risk_level: str,
        risk_factors: list[Dict[str, Any]],
        protective_factors: list[Dict[str, Any]],
       ) -> Dict[str, Any]:
        """
        Build a stakeholder-friendly summary of the model assessment.

        The recommended action reflects the model output only and does not
        replace business rules or analyst judgement.
        """

        classified_as_fraud = probability >= threshold

        
        # Assessment headline

        if classified_as_fraud:
            if risk_level == "high":
                headline = "High Risk — Review Recommended"
            else:
                headline = "Elevated Risk — Review Recommended"
        else:
            if risk_level == "low":
                headline = "Low Risk — No Model Escalation"
            else:
                headline = "Moderate Risk — Below Review Threshold"

        # Main assessment

        if classified_as_fraud:
            assessment = (
                f"The transaction received a fraud probability of "
                f"{probability:.1%}, which exceeds the configured "
                f"{threshold:.0%} fraud-classification threshold."
            )
        else:
            assessment = (
                f"The transaction received a fraud probability of "
                f"{probability:.1%}, which remains below the configured "
                f"{threshold:.0%} fraud-classification threshold."
            )

    
        # Key drivers


        risk_names = [
            factor["display_name"]
            for factor in risk_factors[:3]
        ]

        protective_names = [
            factor["display_name"]
            for factor in protective_factors[:2]
        ]

        drivers = ""

        if risk_names:
            drivers += (
                "The strongest signals increasing the model's risk "
                f"assessment were {self._join_names(risk_names)}."
            )

        if protective_names:
            if drivers:
                drivers += " "

            drivers += (
                "These were partially offset by "
                f"{self._join_names(protective_names)}, which moved the "
                "assessment toward lower risk."
            )

        # Data-quality observation

        all_displayed_factors = (
            risk_factors + protective_factors
        )

        missing_count = sum(
            factor.get("value_status") == "missing"
            for factor in all_displayed_factors
        )

        total_count = len(all_displayed_factors)

        if total_count:
            missing_ratio = missing_count / total_count
        else:
            missing_ratio = 0.0

        if missing_ratio >= 0.5:
            data_quality = (
                "Several influential model inputs were unavailable for this "
                "transaction. The model can still produce an assessment, but "
                "the explanation should be interpreted with additional caution."
            )
        elif missing_count > 0:
            data_quality = (
                "Some influential model inputs were unavailable for this "
                "transaction, although sufficient information remained for "
                "the model to produce an assessment."
            )
        else:
            data_quality = (
                "The influential features shown in this explanation were "
                "available for this transaction."
            )

        # Suggested action


        if classified_as_fraud:
            action = (
                "Consider routing this transaction for additional review "
                "in accordance with the organisation's fraud policies and "
                "business rules."
            )
        elif risk_level == "medium":
            action = (
                "No model-based fraud escalation is currently triggered, "
                "although the transaction may warrant monitoring or review "
                "if additional risk indicators are present."
            )
        else:
            action = (
                "No immediate fraud escalation is indicated by the model. "
                "Normal processing may continue unless other business rules "
                "or external evidence indicate otherwise."
            )

        return {
            "headline": headline,
            "assessment": assessment,
            "key_drivers": drivers,
            "data_quality": data_quality,
            "suggested_action": action,
        }

    def explain_prediction(
            self,
            X: pd.DataFrame,
            predicted_probability: float,
            threshold: float,
            max_features: int = 6,
        ) -> Dict[str, Any]:
            """
            Generate a local explanation using XGBoost's native TreeSHAP
            contribution calculation.

            The final contribution returned by XGBoost is the model baseline.
            All preceding contributions correspond to input features.
            """
            self._ensure_loaded()

            if len(X) != 1:
                raise ValueError(
                    "Local explanations require exactly one transaction."
                )

            booster = self._model.get_booster()

            dmatrix = xgb.DMatrix(
                X,
                feature_names=list(X.columns),
            )

            contribution_matrix = booster.predict(
                dmatrix,
                pred_contribs=True,
            )

            if contribution_matrix.ndim != 2:
                raise RuntimeError(
                    "Unexpected XGBoost contribution output shape: "
                    f"{contribution_matrix.shape}"
                )

            row_contributions = contribution_matrix[0]

            # One contribution per feature, followed by the baseline/bias.
            shap_values = row_contributions[:-1]
            base_value = float(row_contributions[-1])

            if len(shap_values) != len(X.columns):
                raise RuntimeError(
                    "The number of SHAP contributions does not match "
                    "the number of model features."
                )

            contributions: list[Dict[str, Any]] = []

            for feature_name, feature_value, shap_value in zip(
                X.columns,
                X.iloc[0].to_numpy(),
                shap_values,
            ):
                shap_value_float = float(shap_value)

                if np.isclose(
                    shap_value_float,
                    0.0,
                    atol=1e-10,
                ):
                    continue

                feature_name_string = str(feature_name)

                value_info = self._format_feature_value(
                    feature_name=feature_name_string,
                    feature_value=feature_value,
                )

                contributions.append(
                    {
                        "feature": feature_name_string,
                        "display_name": self._get_display_name(
                            feature_name_string
                        ),
                        "feature_group": self._get_feature_group(
                            feature_name_string
                        ),
                        "business_description": (
                            self._get_business_description(
                                feature_name_string
                            )
                        ),
                        "model_value": value_info["model_value"],
                        "display_value": value_info["display_value"],
                        "value_status": value_info["value_status"],
                        "shap_value": shap_value_float,
                        "absolute_impact": abs(shap_value_float),
                        "direction": (
                            "increases_risk"
                            if shap_value_float > 0
                            else "decreases_risk"
                        ),
                        "description": self._describe_feature_impact(
                            feature_name=feature_name_string,
                            feature_value=feature_value,
                            shap_value=shap_value_float,
                        ),
                    }
                )

            contributions.sort(
                key=lambda item: item["absolute_impact"],
                reverse=True,
            )
                        # Select the most influential features for the waterfall chart.
            waterfall_factors = contributions[:max_features]

            waterfall_feature_names = {
                factor["feature"]
                for factor in waterfall_factors
            }

            # Aggregate every contribution that is not individually displayed.
            other_features_contribution = sum(
                factor["shap_value"]
                for factor in contributions
                if factor["feature"] not in waterfall_feature_names
            )

            # TreeSHAP contributions operate in the model's raw score space.
            final_raw_score = (
                base_value
                + sum(float(value) for value in shap_values)
            )

            risk_factors = [
                item
                for item in contributions
                if item["shap_value"] > 0
            ][:max_features]

            protective_factors = [
                item
                for item in contributions
                if item["shap_value"] < 0
            ][:max_features]
            executive_summary = self._build_executive_summary(
                probability=predicted_probability,
                threshold=threshold,
                risk_level=self._get_risk_level(
                    predicted_probability
                ),
                risk_factors=risk_factors,
                protective_factors=protective_factors,
            )

            return {
                "method": "XGBoost TreeSHAP",
                "output_space": "raw_model_score",
                "base_value": base_value,
                "predicted_probability": float(predicted_probability),
                "waterfall": {
                    "base_value": base_value,
                    "final_raw_score": final_raw_score,
                    "other_features_contribution": float(
                        other_features_contribution
                    ),
                    "factors": waterfall_factors,
                },
                "top_risk_factors": risk_factors,
                "top_protective_factors": protective_factors,
                "summary": self._build_explanation_summary(
                    probability=predicted_probability,
                    threshold= threshold,
                    risk_factors=risk_factors,
                    protective_factors=protective_factors,
                ),
                "executive_summary": executive_summary,
                "disclaimer": (
                    "This explanation describes how the model reached its "
                    "prediction. It does not prove that the transaction is "
                    "fraudulent and should support, not replace, investigation."
                ),
            }
    

    @staticmethod
    def _get_risk_level(
        probability: float,
    ) -> str:
        """
        Convert probability into a stakeholder-friendly risk band.

        These are display bands and are separate from the decision
        threshold supplied by the caller.
        """
        if probability >= 0.80:
            return "high"

        if probability >= 0.50:
            return "medium"

        return "low"
    @staticmethod
    def _get_feature_group(feature_name: str) -> str:
        if feature_name == "TransactionAmt":
            return "Transaction details"

        if feature_name.startswith("card"):
            return "Payment card"

        if feature_name.startswith("addr"):
            return "Billing information"

        if "emaildomain" in feature_name:
            return "Email information"

        if feature_name.startswith("C") and feature_name[1:].isdigit():
            return "Anonymised count signals"

        if feature_name.startswith("D") and feature_name[1:].isdigit():
            return "Transaction timing"

        if feature_name.startswith("M") and feature_name[1:].isdigit():
            return "Attribute matching"

        if feature_name.startswith("V") and feature_name[1:].isdigit():
            return "Anonymised behaviour signals"

        if feature_name.startswith("id_"):
            return "Identity information"

        if "_FE" in feature_name:
            return "Historical familiarity"

        if "_mean" in feature_name or "_std" in feature_name:
            return "Historical transaction behaviour"

        return "Other model signals"

    @staticmethod
    def _json_safe_value(
        value: Any,
    ) -> Any:
        """
        Convert NumPy and pandas values into JSON-safe Python values.
        """
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        return value

    @staticmethod
    def _get_display_name(feature_name: str) -> str:
        """
        Convert internal model feature names into conservative,
        stakeholder-friendly labels.
        """
        if feature_name in FEATURE_DISPLAY_NAMES:
            return FEATURE_DISPLAY_NAMES[feature_name]

        # C1-C14 are anonymised count-related variables.
        if feature_name.startswith("C") and feature_name[1:].isdigit():
            return (
                f"Anonymised transaction count signal "
                f"{feature_name[1:]}"
            )

        # D1-D15 are anonymised time-difference variables.
        if feature_name.startswith("D") and feature_name[1:].isdigit():
            return (
                f"Anonymised transaction timing signal "
                f"{feature_name[1:]}"
            )

        # M1-M9 are anonymised matching indicators.
        if feature_name.startswith("M") and feature_name[1:].isdigit():
            return (
                f"Anonymised transaction match signal "
                f"{feature_name[1:]}"
            )

        # V1-V339 are anonymised engineered transaction variables.
        if feature_name.startswith("V") and feature_name[1:].isdigit():
            return (
                f"Anonymised transaction behaviour signal "
                f"{feature_name[1:]}"
            )

        # id_01-id_38 are anonymised identity-related variables.
        if feature_name.startswith("id_"):
            identity_number = feature_name.removeprefix("id_")
            return f"Anonymised identity signal {identity_number}"

        readable_name = feature_name.replace("|", " and ")
        readable_name = readable_name.replace("_", " ")
        readable_name = " ".join(readable_name.split())

        return readable_name.capitalize()

    @staticmethod
    def _get_business_description(feature_name: str) -> str:
        """
        Describe what a feature represents without claiming knowledge
        that the anonymised dataset does not provide.
        """
        if feature_name in FEATURE_BUSINESS_DESCRIPTIONS:
            return FEATURE_BUSINESS_DESCRIPTIONS[feature_name]

        if feature_name.startswith("C") and feature_name[1:].isdigit():
            return (
                "An anonymised count-related transaction indicator. "
                "The dataset does not disclose its exact business definition."
            )

        if feature_name.startswith("D") and feature_name[1:].isdigit():
            return (
                "An anonymised time-difference indicator associated with "
                "the transaction. Its exact definition is not disclosed."
            )

        if feature_name.startswith("M") and feature_name[1:].isdigit():
            return (
                "An anonymised matching indicator comparing transaction "
                "attributes. Its exact comparison is not disclosed."
            )

        if feature_name.startswith("V") and feature_name[1:].isdigit():
            return (
                "An anonymised engineered transaction-behaviour indicator. "
                "Its exact calculation is not disclosed."
            )

        if feature_name.startswith("id_"):
            return (
                "An anonymised identity-related attribute available for "
                "some transactions. Its exact definition is not disclosed."
            )

        return (
            "An engineered input used by the fraud model to assess "
            "this transaction."
        )
    @staticmethod
    def _format_feature_value(
        feature_name: str,
        feature_value: Any,
    ) -> Dict[str, Any]:
        """
        Return both the model value and a stakeholder-facing representation.

        The model value is retained for technical auditability, while the
        display value hides preprocessing sentinels such as -1.
        """
        json_value = FraudPredictor._json_safe_value(feature_value)

        if json_value is None or json_value == -1:
            return {
                "model_value": json_value,
                "display_value": "Not available",
                "value_status": "missing",
            }
        if feature_name == "TransactionAmt":
            return {
                "model_value": json_value,
                "display_value": f"{float(json_value):,.2f}",
                "value_status": "available",
            }

        if isinstance(json_value, float):
            if feature_name.endswith("_FE"):
                return {
                    "model_value": json_value,
                    "display_value": f"{json_value:.4%}",
                    "value_status": "available",
                }

            return {
                "model_value": json_value,
                "display_value": f"{json_value:,.3f}",
                "value_status": "available",
            }

        return {
            "model_value": json_value,
            "display_value": str(json_value),
            "value_status": "available",
        }

    
    def _describe_feature_impact(
        self,
        feature_name: str,
        shap_value: float,
        feature_value: Any,
    ) -> str:
        """
        Explain what the model did without treating association as causation.
        """
        display_name = self._get_display_name(feature_name)

        value_info = self._format_feature_value(
            feature_name=feature_name,
            feature_value=feature_value,
        )

        if value_info["value_status"] == "missing":
            value_context = (
                "Information for this feature was not available, and the "
                "model used its configured missing-value representation."
            )
        else:
            value_context = (
                f"The value presented to the model was "
                f"{value_info['display_value']}."
            )

        if shap_value > 0:
            direction_text = (
                "For this transaction, this signal moved the model toward "
                "a higher fraud-risk assessment."
            )
        else:
            direction_text = (
                "For this transaction, this signal moved the model toward "
                "a lower fraud-risk assessment."
            )

        return (
            f"{display_name}: {direction_text} {value_context}"
        )

    def _build_explanation_summary(
        self,
        probability: float,
        threshold: float,
        risk_factors: list[Dict[str, Any]],
        protective_factors: list[Dict[str, Any]],
    ) -> str:
        """
        Produce a plain-English summary suitable for a fraud analyst.
        """
        risk_level = self._get_risk_level(probability)
        percentage = probability * 100

        summary = (
            f"The model estimated a {percentage:.1f}% probability of fraud, "
            f"which falls within the {risk_level}-risk band."
        )

        if risk_factors:
            risk_names = [
                factor["display_name"]
                for factor in risk_factors[:3]
            ]

            summary += (
                " The strongest signals increasing the model's risk "
                f"assessment were {self._join_names(risk_names)}."
            )

        if protective_factors:
            protective_names = [
                factor["display_name"]
                for factor in protective_factors[:2]
            ]

            summary += (
                " These were partly offset by "
                f"{self._join_names(protective_names)}, which moved the "
                "assessment toward lower risk."
            )

        if probability < threshold:
            summary += (
                 f" Overall, the score remained below the configured "
                 f"{threshold:.0%} fraud-classification threshold."
            )
        else:
            summary += (
                f" Overall, the score exceeded the configured "
                f"{threshold:.0%} fraud-classification threshold and may "
                "warrant further review."
            )

        return summary

    @staticmethod
    def _join_names(
        names: list[str],
    ) -> str:
        if not names:
            return ""

        if len(names) == 1:
            return names[0]

        if len(names) == 2:
            return f"{names[0]} and {names[1]}"

        return f"{', '.join(names[:-1])}, and {names[-1]}"