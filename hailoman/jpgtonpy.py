import os
import cv2
import numpy as np

# Directory containing your JPEG images
image_dir = 'calib-dataset'

# Directory to save the .npy files
output_dir = 'calib-dataset-npy'
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the image directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # Construct full file path
        file_path = os.path.join(image_dir, filename)
        
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        
        # Check if the image was successfully loaded
        if image is None:
            print(f"Warning: {file_path} could not be loaded.")
            continue
        
        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Optionally, preprocess the image (e.g., resizing, normalization)
        # Example: Resize to 224x224
        image_rgb = cv2.resize(image_rgb, (224, 224))
        
        # Convert the image to a NumPy array
        image_array = np.array(image_rgb)
        
        # Define the output .npy file path
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(output_dir, npy_filename)
        
        # Save the NumPy array as a .npy file
        np.save(npy_path, image_array)
        
        print(f"Saved {npy_path}")

print("Conversion complete.")

