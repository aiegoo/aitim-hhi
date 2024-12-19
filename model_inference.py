import torch
from torchvision import transforms, models
from PIL import Image

# Define the number of classes (should match your training setup)
num_classes = 5

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

# Load and preprocess the image
image_path = "image_7.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)

# Define the class names (should match the classes used during training)
class_names = ['background', 'hammer', 'measuring_tape', 'pliers', 'screwdriver']

# Print the predicted class
print(f"Predicted class: {class_names[preds.item()]}")
