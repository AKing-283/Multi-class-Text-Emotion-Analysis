import streamlit as st
import joblib
import random

with open('emotion_pipeline_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

emotion_mapping = {
    0: 'happy',
    1: 'sad',
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
    return random.choice(responses[emotion])

# Streamlit UI
st.title("🗣️ Emotion-Based Chatbot")
st.write("Type a message and let the chatbot detect the emotion and respond.")

user_input = st.text_input("You:", "")

if user_input:
    predicted_label = model.predict([user_input])[0]  # Predicts emotion

    # Map prediction to emotion (defaults to 'neutral' if not found)
    predicted_emotion = emotion_mapping.get(predicted_label, 'neutral')

    # Debugging confidence scores (if available)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba([user_input])[0]
        emotion_labels = [emotion_mapping.get(i, f"Unknown({i})") for i in range(len(probas))]
        st.write("**Prediction Confidence:**", dict(zip(emotion_labels, probas)))

    response = get_emotion_response(predicted_emotion)

    st.write(f"**Emotion Detected:** {predicted_emotion.capitalize()}")
    st.write(f"🤖 **Chatbot:** {response}")

# ----------------------------
# 🔹 TEST CASES (Uncomment to test)
# ----------------------------
# if st.button("Run Test Cases"):
#     test_inputs = [
#         "I just got a promotion at work! 🎉",  # Expected: Happy
#         "I feel so lonely right now...",  # Expected: Sad
#         "Why do people never listen to me? 😡",  # Expected: Angry
#         "It looks like it might rain today.",  # Expected: Neutral
#         "I am feeling extremely joyful today!",  # Expected: Happy
#         "I'm really upset about what happened.",  # Expected: Sad
#         "This is so frustrating! I can't believe this.",  # Expected: Angry
#         "Everything is okay, I guess.",  # Expected: Neutral
#     ]
#     
#     for text in test_inputs:
#         emotion_pred = model.predict([text])[0]
#         mapped_emotion = emotion_mapping.get(emotion_pred, 'neutral')
#         st.write(f"**Input:** {text} → **Predicted Emotion:** {mapped_emotion.capitalize()}")

