from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DiseaseModel:
    """Multi-disease classifier trained on symptoms CSV.

    Expected CSV: symptoms columns (0/1) + final column "prognosis" with disease label.
    File name: disease_symptoms.csv in data/ directory.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        self._train()

    def _load_csv(self) -> pd.DataFrame:
        path = self.data_dir / "disease_symptoms.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Please save your symptoms CSV as 'disease_symptoms.csv' in {self.data_dir}"
            )
        # The user may have tabs; let pandas auto-detect separator
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep="\t")
        return df

    def _train(self) -> None:
        df = self._load_csv()
        # normalize column names: strip spaces
        df.columns = [c.strip() for c in df.columns]
        # last column is target
        self.symptom_features: List[str] = [c for c in df.columns if c.lower() != "prognosis"]
        X = df[self.symptom_features]
        y = df[[c for c in df.columns if c.lower() == "prognosis"][0]]

        n_samples = len(df)
        if n_samples < 5:
            # dataset too small; train on all and evaluate on train
            self.model.fit(X, y.values.ravel())
            self.accuracy = float(accuracy_score(y, self.model.predict(X)))
        else:
            # disable stratify if any class has < 2 samples
            vc = y.value_counts()
            stratify_y = y if (vc.min() >= 2) else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )
            self.model.fit(X_train, y_train.values.ravel())
            self.accuracy = float(accuracy_score(y_test, self.model.predict(X_test)))

        # Keep class names
        self.class_names = list(self.model.classes_)

    def predict_from_symptoms(self, selected_symptoms: List[str]) -> Dict[str, Any]:
        # Build a single-row feature vector of 0/1 given selected symptoms
        row = {f: 1 if f in selected_symptoms else 0 for f in self.symptom_features}
        X = pd.DataFrame([row])[self.symptom_features]
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        top = sorted(
            [{"disease": d, "probability": float(p)} for d, p in zip(self.class_names, proba)],
            key=lambda x: x["probability"], reverse=True
        )[:5]
        return {
            "prediction": str(pred),
            "top_candidates": top,
            "model_accuracy": self.accuracy,
            "selected_symptoms": selected_symptoms,
        }


