import argparse
import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType)

# Set up argument parser
parser = argparse.ArgumentParser(description='Perform image recognition using a HEF file.')
parser.add_argument('image_path', type=str, help='Path to the input image file')
args = parser.parse_args()

# Load the HEF file
hef_path = 'resnet18.hef'
hef = HEF(hef_path)

# Create a virtual device
with VDevice() as vdevice:
    # Configure the network group
    configure_params = ConfigureParams.create_from_hef(hef)
    network_group = vdevice.configure(hef, configure_params)[0]

    # Create input and output virtual streams
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        # Read and preprocess the input image
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Image at path '{args.image_path}' could not be loaded. Please check the file path.")
        input_shape = input_vstreams_params[0].shape  # Assuming a single input
        resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize to match model's input
        input_data = np.expand_dims(resized_image, axis=0)  # Add batch dimension

        # Perform inference
        output = infer_pipeline.infer(input_data)

        # Process the output as needed
        # For example, if it's a classification model:
        predictions = output[0]  # Assuming a single output
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        print(f'Predicted class: {predicted_class} with confidence {confidence:.2f}')

