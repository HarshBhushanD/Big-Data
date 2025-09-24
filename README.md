# Disease & Diabetes Detection (Flask)

## Setup
1. Install Python 3.10+.
2. In a terminal from this folder:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Provide your CSV data
Create a `data/` directory and place the CSV files:
- `data/pima_diabetes.csv` — columns: `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`
- `data/disease_symptoms.csv` — symptoms (0/1) columns + final `prognosis` column.

Note: If your disease CSV is tab-separated, it is auto-detected.

## Run the app
```
python app.py
```
Open `http://localhost:5000`.

## Features
- Email/password signup/login using SQLite.
- Diabetes risk prediction using Pima features.
- Multi-disease prediction from symptom selections.

## Optional: Firebase Auth
You can swap SQLite auth for Firebase later. For now, environment variable `FLASK_SECRET` sets Flask session secret.


