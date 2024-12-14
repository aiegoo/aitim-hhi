import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('fine_tuned_efficientnet_model.h5')  # Or your best model

# Define train_generator (same as in your training script)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset',  # Path to your dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to get consistent class indices
)

# Get class labels (assuming they are the same as folder names)
class_labels = list(train_generator.class_indices.keys())

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while(True):
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    input_tensor = tf.keras.applications.efficientnet.preprocess_input(resized_frame)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Make a prediction
    prediction = model.predict(input_tensor)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {predicted_class} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
