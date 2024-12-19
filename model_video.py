import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Function to load class names from labels.txt
def load_class_names(labels_path):
    try:
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        print(f"Error: {labels_path} not found.")
        exit()

# Path to labels.txt
labels_path = "tool_dataset/labels.txt"
class_names = load_class_names(labels_path)

# Define the number of classes (should match the number of labels in labels.txt)
num_classes = len(class_names)

# Load the model architecture (ResNet18 in this case)
model = models.resnet18(pretrained=False)  # Set pretrained=False because you're loading your weights
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load("tool_classifier.pth"))
model.eval()

# Define the device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to PIL format for processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_names[preds.item()]

    # Display the result on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
