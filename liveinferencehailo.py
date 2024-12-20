import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType)

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
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Preprocess the frame
            input_shape = input_vstreams_params[0].shape  # Assuming a single input
            resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))  # Resize to match model's input
            input_data = np.expand_dims(resized_frame, axis=0)  # Add batch dimension

            # Perform inference
            output = infer_pipeline.infer(input_data)

            # Process the output as needed
            # For example, if it's a classification model:
            predictions = output[0]  # Assuming a single output
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]

            # Display the results
            cv2.putText(frame, f'Class: {predicted_class}, Confidence: {confidence:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam Inference', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

