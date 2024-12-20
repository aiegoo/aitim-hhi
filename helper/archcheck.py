import torch

# Load the .pth file
model_data = torch.load("../tool_classifier.pth")

# Inspect the contents
print(type(model_data))
print(model_data.keys() if isinstance(model_data, dict) else "Not a dictionary")

