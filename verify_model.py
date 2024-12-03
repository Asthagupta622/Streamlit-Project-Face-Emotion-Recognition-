from tensorflow.keras.models import load_model

# Load the existing H5 model
model = load_model('emotion.h5')

# Re-save the model to ensure it's not corrupted
model.save('emotion.h5')  # Overwrite the existing file if needed

# Convert the model to TensorFlow's SavedModel format
model.save('emotion_model', save_format='tf')
print("Model saved successfully in both H5 and TensorFlow SavedModel format.")
