from flask import Flask, render_template, request
import joblib
import os
import random
import tweepy
from dotenv import load_dotenv
import os
from flask_limiter import Limiter

load_dotenv()
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

def get_twitter_client():
    return tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

app = Flask(__name__)

# Load the trained emotion detection model safely
model_path = os.path.join(os.path.dirname(__file__), "emotion_pipeline_model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model is missing

# Define emotion mapping with corresponding emojis
emotion_mapping = {
    1: ("happy", "😊"),
    2: ("sad", "😢"),
    3: ("angry", "😡"),
    0: ("neutral", "😐")
}

# Function to get chatbot responses
def get_emotion_response(emotion):
    responses = {
        "happy": [
            "I'm glad you're feeling happy! Keep spreading positivity! 😊",
            "Happiness is contagious! Keep smiling! 😃",
            "Enjoy the moment! Life is beautiful. 🌟"
        ],
        "sad": [
            "I'm here for you. Remember, tough times don't last. 💙",
            "It's okay to feel sad sometimes. You're not alone. 🤗",
            "Try to do something you enjoy—it might lift your mood! ☀️"
        ],
        "angry": [
            "Take a deep breath. Maybe a short walk can help calm your mind. 🌿",
            "I understand anger can be tough. Try writing your thoughts down. ✍️",
            "Listening to calming music might help. Stay strong! 🎵"
        ],
        "neutral": [
            "Got it! Let me know if I can assist you with anything. 🙂",
            "Neutral is good. How’s your day going? ☕",
            "Would you like to talk about something fun? 🎉"
        ]
    }
    return random.choice(responses.get(emotion, ["I'm here to help, no matter what you're feeling! 💜"]))

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
            predicted_emotion, emoji = emotion_mapping.get(int(predicted_label), ("neutral", "🤔"))
            response = get_emotion_response(predicted_emotion)
        except Exception as e:
            return render_template("error.html", message=f"Prediction failed: {str(e)}")

        return render_template("result.html", user_input=user_input, emotion=predicted_emotion, emoji=emoji, response=response)
    
@app.route("/analyze_tweets", methods=["GET", "POST"])
def analyze_tweets():
    if request.method == "POST":
        search_query = request.form.get("search_query", "")
        if not search_query:
            return render_template("error.html", message="Please enter a search term.")
        
        try:
            client = get_twitter_client()
            tweets = client.search_recent_tweets(
                query=search_query,
                max_results=50,
                tweet_fields=["created_at"]
            )
            
            if not tweets.data:
                return render_template("error.html", message="No tweets found.")
            
            processed_tweets = []
            for tweet in tweets.data:
                text = tweet.text
                prediction = model.predict([text])[0]
                emotion, emoji = emotion_mapping.get(int(prediction), ("neutral", "🤔"))
                processed_tweets.append({
                    "text": text,
                    "emotion": emotion,
                    "emoji": emoji
                })
            
            return render_template("tweet_results.html", 
                                 tweets=processed_tweets,
                                 search_term=search_query)
            
        except Exception as e:
            return render_template("error.html", message=f"Twitter error: {str(e)}")
    
    return render_template("tweet_search.html")

limiter = Limiter(app=app, key_func=lambda: request.remote_addr)
limiter.limit("10 per minute")(analyze_tweets)

if __name__ == "__main__":
    app.run(debug=True)
