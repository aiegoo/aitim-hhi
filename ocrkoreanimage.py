import cv2
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # Add more languages if needed

# Load the image
image_path = "KakaoTalk_20241202_170309467.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f"Image at '{image_path}' not found or could not be loaded.")

# Load a font that supports Korean
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Change this to a valid font path on your system
font = ImageFont.truetype(font_path, 24)

# Perform OCR on the frame
results = ocr.ocr(frame, cls=True)

# Convert the frame to a PIL image for drawing text
frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(frame_pil)

# Prepare to save the recognized text
output_text_path = "recognized_text.txt"
with open(output_text_path, "w", encoding="utf-8") as text_file:
    # Check if OCR detected any text
    if results and results[0]:
        for line in results[0]:  # Iterate over detected text
            if len(line) >= 2:  # Ensure line has at least bbox and text
                bbox, text = line[:2]  # Extract bounding box and text
                confidence = line[2] if len(line) > 2 else 1.0  # Default confidence to 1.0 if missing

                # Write the text to the file
                text_file.write(f"{text}\n")

                # Extract coordinates
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                # Draw rectangle around detected text
                draw.rectangle([top_left, bottom_right], outline="green", width=2)

                # Display text and confidence score
                display_text = f"{text} ({confidence:.2f})"
                draw.text((top_left[0], top_left[1] - 30), display_text, font=font, fill="yellow")

# Convert the PIL image back to OpenCV format
result_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Save the processed image
output_image_path = "output_image_with_ocr.jpg"  # Define the path to save the result
cv2.imwrite(output_image_path, result_frame)
print(f"Resulting image saved at {output_image_path}")
print(f"Recognized text saved at {output_text_path}")

# Optionally display the processed image
cv2.imshow('PaddleOCR Image', result_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
