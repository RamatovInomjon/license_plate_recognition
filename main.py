import cv2
from ultralytics import YOLO


# Map character classes to actual characters
character_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                    'U', 'V', 'W', 'X', 'Y', 'Z']

# Load YOLOv8 models
license_plate_model = YOLO("weights/detection.pt")  # Replace with your license plate model
character_model = YOLO("weights/recognition.pt")          # Replace with your character model

# Open video or camera
cap = cv2.VideoCapture('test.mp4')  # or 0 for webcam

# open if you want to record results
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (e.g., 'XVID' for .avi, 'mp4v' for .mp4)
# out = cv2.VideoWriter('results.mp4', fourcc, 30.0, 
                    #   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect license plates in the frame
    license_plates = license_plate_model(frame)
    for plate in license_plates[0].boxes:
        x1, y1, x2, y2 = map(int, plate.xyxy[0])  # Get bounding box coordinates
        cropped_license_plate = frame[y1:y2, x1:x2]

        # Step 2: Detect characters in the cropped license plate
        characters = character_model(cropped_license_plate)

        # Collect detected characters and their x-coordinates
        detected_characters = []
        for char in characters[0].boxes:
            x1_char, y1_char, x2_char, y2_char = map(int, char.xyxy[0])
            character_class = int(char.cls[0])  # Detected character class index
            detected_characters.append((character_class, x1_char))

        # Sort characters by x-coordinate
        detected_characters.sort(key=lambda char: char[1])
        
        license_text = ''.join([character_map[char[0]] for char in detected_characters])

        # Print the result
        print("Detected License Plate Text:", license_text)
        
        # Draw bounding box and text on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, license_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("License Plate Recognition", frame)
    # open if you want to record results
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
