from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained emotion detection model safely
model_path = os.path.join(os.path.dirname(__file__), "emotion_pipeline_model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model is missing

# Define emotion mapping
emotion_mapping = {
    1: "happy",
    2: "sad",
    3: "love",
    4: "Anger",
    0: "neutral"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if not model:
            return render_template("error.html", message="Model not found or failed to load.")
        
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return render_template("error.html", message="Please enter some text.")
        
        try:
            predicted_label = model.predict([user_input])[0]  # Predict emotion
            predicted_emotion = emotion_mapping.get(int(predicted_label), "neutral")
        except Exception as e:
            return render_template("error.html", message=f"Prediction failed: {str(e)}")

        return render_template("result.html", user_input=user_input, emotion=predicted_emotion)

if __name__ == "__main__":
    app.run(debug=True)
