import onnx

# Load the ONNX model
model = onnx.load('resnet18.onnx')

# Print input node names
print("Input Nodes:")
for input in model.graph.input:
    print(input.name)

# Print output node names
print("\nOutput Nodes:")
for output in model.graph.output:
    print(output.name)

