from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class DiabetesModel:
    """Diabetes prediction model that loads a CSV from data/.

    Expected CSV columns:
    Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.feature_names: List[str] = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self._train()

    def _load_csv(self) -> pd.DataFrame:
        # look for file named pima_diabetes.csv
        csv_path = self.data_dir / "pima_diabetes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Please place your diabetes CSV as 'pima_diabetes.csv' in {self.data_dir}"
            )
        df = pd.read_csv(csv_path)
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # replace zeros in medical fields with median
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for c in zero_cols:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan)
                df[c] = df[c].fillna(df[c].median())
        return df

    def _train(self) -> None:
        df = self._load_csv()
        df = self._preprocess(df)
        X = df[self.feature_names]
        y = df["Outcome"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        self.accuracy = float(acc)

    def default_input_example(self) -> Dict[str, float]:
        return {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 85,
            "BMI": 28.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35,
        }

    def predict(self, user_values: Dict[str, Any]) -> Dict[str, Any]:
        row = {k: float(user_values.get(k, 0.0)) for k in self.feature_names}
        X = pd.DataFrame([row])[self.feature_names]
        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0].tolist()
        return {
            "prediction": pred,
            "probabilities": {"no_diabetes": proba[0], "diabetes": proba[1]},
            "model_accuracy": self.accuracy,
        }


