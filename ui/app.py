from flask import Flask, render_template, request, session, redirect, url_for
from joblib import load
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess import preprocess_text, preprocess_series

app = Flask(__name__)
app.secret_key = "supersecretkey"

pipeline = load("artifacts/model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    text_input = ""

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        text_input = request.form.get("text", "").strip()

        if text_input:
            prediction = pipeline.predict([text_input])[0]

            prob_dict = None
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                prob = pipeline.predict_proba([text_input])[0]
                prob_dict = {cls: f"{p*100:.1f}%" for cls, p in zip(pipeline.classes_, prob)}

            session["history"].insert(0, {
                "text": text_input,
                "prediction": prediction
            })
            session.modified = True

            probability = prob_dict

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        text=text_input,
        history=session["history"]
    )

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["history"] = []
    session.modified = True
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
