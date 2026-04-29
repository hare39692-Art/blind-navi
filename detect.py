from ultralytics import YOLO
import pyttsx3

# Load the YOLOv8 nano model (downloads automatically on first run)
model = YOLO("yolov8n.pt")

# Run detection on your test image
print("Running detection...")
results = model("test.jpg")
result = results[0]

# Extract detected object names
detected = []
for box in result.boxes:
    class_id = int(box.cls)
    name = result.names[class_id]
    confidence = round(float(box.conf) * 100)
    detected.append(name)
    print(f"  Found: {name} ({confidence}% confident)")

# Remove duplicate names
unique_objects = list(set(detected))

# Build the spoken sentence
if unique_objects:
    joined = ", ".join(unique_objects)
    sentence = f"I see {len(unique_objects)} objects: {joined}"
else:
    sentence = "Nothing detected in front of you."

print(f"\nSpeaking: {sentence}")

# Speak it out loud
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say(sentence)
engine.runAndWait()

# Save result image with bounding boxes drawn
result.save(filename="result.jpg")
print("Saved result.jpg — open it to see the detected boxes!")