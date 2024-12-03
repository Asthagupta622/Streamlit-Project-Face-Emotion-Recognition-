import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model("emotion.h5")  # Make sure "emotion.h5" is in the same directory or provide a correct path

# Define emotion labels (update as per your model's classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit UI
st.title("Real-Time Emotion Detection")
FRAME_WINDOW = st.image([])
st.text("Press 'q' to exit the app.")

# Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not access the camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame. Check your camera.")
            break

        # Process the frame for emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (48, 48))  # Match the model's input size
        reshaped_frame = np.expand_dims(np.expand_dims(resized_frame, -1), 0) / 255.0  # Normalize pixel values
        
        # Predict the emotion
        predictions = model.predict(reshaped_frame)
        emotion = emotion_labels[np.argmax(predictions)]
        
        # Display the emotion label on the video frame
        cv2.putText(frame, emotion, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the processed frame in the Streamlit app
        FRAME_WINDOW.image(frame, channels="BGR")
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
