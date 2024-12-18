import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize EasyOCR reader for Korean and English
reader = easyocr.Reader(['ko'])  # 'ko' for Korean, 'en' for English

# Load the NanumGothic font
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Update if needed
font = ImageFont.truetype(font_path, 24)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame skipping parameters
frame_count = 0
ocr_interval = 10  # Perform OCR every 10 frames

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform OCR only on every n-th frame
    if frame_count % ocr_interval == 0:
        # Downscale frame for faster processing
        resized_frame = cv2.resize(frame, (320, 240))  # Reduce size
        results = reader.readtext(resized_frame)

        # Scale bounding boxes back to original frame size
        scale_x = frame.shape[1] / resized_frame.shape[1]
        scale_y = frame.shape[0] / resized_frame.shape[0]
        scaled_results = []
        for (bbox, text, confidence) in results:
            scaled_bbox = [(int(x * scale_x), int(y * scale_y)) for (x, y) in bbox]
            scaled_results.append((scaled_bbox, text, confidence))
        results = scaled_results

    # Convert frame to PIL image for rendering text
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Draw the OCR results
    if 'results' in locals():  # Ensure results exist
        for (bbox, text, confidence) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw rectangle around detected text
            draw.rectangle([top_left, bottom_right], outline="green", width=2)

            # Display the detected text with Pillow
            display_text = f"{text} ({confidence:.2f})"
            draw.text((top_left[0], top_left[1] - 30), display_text, font=font, fill="yellow")

    # Convert the PIL image back to OpenCV format
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Show the frame
    cv2.imshow('Optimized EasyOCR Webcam with Korean Support', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

