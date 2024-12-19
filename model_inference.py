import torch
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Load class names from labels.txt
labels_file = "tool_dataset/labels.txt"
with open(labels_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define the number of classes (should match your training setup)
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

# Open a file dialog to select the image file
root = tk.Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if not image_path:
    print("No file selected. Exiting...")
    exit()

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)

# Print the predicted class
print(f"Predicted class: {class_names[preds.item()]}")
