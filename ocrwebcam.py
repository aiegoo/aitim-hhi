import cv2
from paddleocr import PaddleOCR

# Initialize PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Add more languages if needed

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform OCR on the frame
    results = ocr.ocr(frame, cls=True)

    # Check if OCR detected any text
    if results and results[0]:
        for line in results[0]:  # Iterate over detected text
            if len(line) >= 2:  # Ensure line has at least bbox and text
                bbox, text = line[:2]  # Extract bounding box and text
                confidence = line[2] if len(line) > 2 else 1.0  # Default confidence to 1.0 if missing

                # Extract coordinates
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                # Draw rectangle around detected text
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Display text and confidence score
                display_text = f"{text} ({confidence:.2f})"
                cv2.putText(frame, display_text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow('PaddleOCR Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

