import streamlit as st
import joblib
import random

# Load the model
with open('emotion_pipeline_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Define emotion mapping
emotion_mapping = {
    1: 'happy',
    0: 'sad',
    2: 'angry',
    3: 'neutral'
}

# Function to get chatbot responses
def get_emotion_response(emotion):
    responses = {
        'happy': [
            "I'm glad you're feeling happy! Keep spreading positivity! 😊",
            "Happiness is contagious! Keep smiling! 😃",
            "Enjoy the moment! Life is beautiful. 🌟"
        ],
        'sad': [
            "I'm here for you. Remember, tough times don’t last. 💙",
            "It's okay to feel sad sometimes. You're not alone. 🤗",
            "Try to do something you enjoy—it might lift your mood! ☀️"
        ],
        'angry': [
            "Take a deep breath. Maybe a short walk can help calm your mind. 🌿",
            "I understand anger can be tough. Try writing your thoughts down. ✍️",
            "Listening to calming music might help. Stay strong! 🎵"
        ],
        'neutral': [
            "Got it! Let me know if I can assist you with anything. 🙂",
            "Neutral is good. How’s your day going? ☕",
            "Would you like to talk about something fun? 🎉"
        ]
    }
    return random.choice(responses.get(emotion, "I'm here to help, no matter what you're feeling! 💜"))

# Streamlit UI
st.title("🗣️ Emotion-Based Chatbot")
st.write("Type a message and let the chatbot detect the emotion and respond.")

user_input = st.text_input("You:", "")

if user_input:
    predicted_label = model.predict([user_input])[0]  # Predicts emotion
    
    # Ensure predicted_label is an integer and map to emotion text
    predicted_emotion = emotion_mapping.get(int(predicted_label), 'neutral')
    
    response = get_emotion_response(predicted_emotion)

    st.write(f"**Emotion Detected:** {predicted_emotion.capitalize()}")
    st.write(f"🤖 **Chatbot:** {response}")
