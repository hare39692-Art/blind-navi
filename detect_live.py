from ultralytics import YOLO
import cv2
import time
import subprocess

# Load model
model = YOLO("yolov8n.pt")

# Speak using Windows PowerShell (no pyttsx3 needed)
def speak(text):
    subprocess.Popen([
        "powershell", "-Command",
        f'Add-Type -AssemblyName System.Speech; '
        f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
    ])

# Open webcam (change 0 to 1 if black screen)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not found! Try changing 0 to 1")
    exit()

print("Blind Assist is running... Press Q to quit")

last_spoken = 0
last_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read from webcam")
        break

    # Run detection
    results = model(frame, verbose=False)
    result = results[0]

    # Get detected object names above 50% confidence
    detected = []
    for box in result.boxes:
        class_id = int(box.cls)
        name = result.names[class_id]
        confidence = float(box.conf)
        if confidence > 0.5:
            detected.append(name)

    unique_objects = list(set(detected))

    # Speak every 4 seconds only if objects changed
    current_time = time.time()
    if unique_objects and (current_time - last_spoken > 4) and (unique_objects != last_objects):
        joined = ", ".join(unique_objects)
        sentence = f"I see {joined}"
        print(f"Speaking: {sentence}")
        speak(sentence)
        last_spoken = current_time
        last_objects = unique_objects

    # Draw bounding boxes
    annotated = result.plot()

    # Show object names on screen
    y = 30
    for obj in unique_objects:
        cv2.putText(annotated, obj, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30

    # Show the live window
    cv2.imshow("Blind Assist - Live Detection", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")