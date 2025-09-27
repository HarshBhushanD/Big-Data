import os
import sqlite3
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Local ML modules
from ml_diabetes import DiabetesModel
from ml_disease import DiseaseModel


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "auth.db"
DATA_DIR = BASE_DIR / "data"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    if not DB_PATH.exists():
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        conn.close()


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

    DATA_DIR.mkdir(exist_ok=True)

    init_db()

    diabetes_model = DiabetesModel(DATA_DIR)
    disease_model = DiseaseModel(DATA_DIR)

    @app.route("/")
    def index():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    # auth
    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not (name and email and password):
                flash("All fields are required.", "error")
                return render_template("signup.html")

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                    (email, name, generate_password_hash(password)),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                flash("Email already registered.", "error")
                conn.close()
                return render_template("signup.html")

            cur.execute("SELECT id, name FROM users WHERE email = ?", (email,))
            user = cur.fetchone()
            conn.close()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            return redirect(url_for("dashboard"))

        return render_template("signup.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cur.fetchone()
            conn.close()
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["name"] = user["name"]
                return redirect(url_for("dashboard"))
            flash("Invalid email or password.", "error")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    # app
    @app.route("/dashboard")
    def dashboard():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return render_template("dashboard.html")

    # diabetes
    @app.route("/predict/diabetes", methods=["GET", "POST"])
    def predict_diabetes():
        if not session.get("user_id"):
            return redirect(url_for("login"))

        if request.method == "POST":
           
            form_values: Dict[str, Any] = {}
            for feature in diabetes_model.feature_names:
                val = request.form.get(feature, "0").strip()
                try:
                    form_values[feature] = float(val)
                except ValueError:
                    form_values[feature] = 0.0

            result = diabetes_model.predict(form_values)
            return render_template(
                "diabetes_result.html",
                result=result,
            )

        return render_template(
            "predict_diabetes.html",
            features=diabetes_model.feature_names,
            defaults=diabetes_model.default_input_example(),
        )

    # other diseases
    @app.route("/predict/disease", methods=["GET", "POST"])
    def predict_disease():
        if not session.get("user_id"):
            return redirect(url_for("login"))

        if request.method == "POST":
            selected = request.form.getlist("symptoms")
            result = disease_model.predict_from_symptoms(selected)
            return render_template("disease_result.html", result=result)

        return render_template(
            "predict_disease.html",
            symptoms=disease_model.symptom_features,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


