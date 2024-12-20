import torch
import torch.nn as nn
import torchvision.models as models

# Step 1: Read the number of classes from labels.txt
labels_file = "tool_dataset/labels.txt"  # Path to your labels.txt file
with open(labels_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
num_classes = len(classes)  # Number of classes is the number of lines in the file

# Step 2: Define the model
# Use torchvision's ResNet and replace the final fully connected layer
model = models.resnet18(weights=None)  # Do not load pretrained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Step 3: Load the state dictionary
state_dict = torch.load("tool_classifier.pth", weights_only=True)  # Set weights_only=True to suppress warnings
model.load_state_dict(state_dict)
model.eval()

# Step 4: Define a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust dimensions to match your input shape

# Step 5: Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "tool_classifier.onnx",  # Output ONNX file
    export_params=True,
    opset_version=11,  # ONNX opset version
    do_constant_folding=True,  # Optimize the model
    input_names=["input"],  # Name of input tensor
    output_names=["output"],  # Name of output tensor
    dynamic_axes={
        "input": {0: "batch_size"},  # Dynamic batch size
        "output": {0: "batch_size"}
    }
)

print(f"ONNX model has been saved as 'tool_classifier.onnx' with {num_classes} classes.")

