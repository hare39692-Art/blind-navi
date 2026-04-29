from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import re
import requests as req
import os

app = Flask(__name__)
CORS(app)

# Load YOLO lazily — only when first request comes in
model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("YOLO model loaded!")
    return model

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImUyZjMzNDkyYzllNzQ4NTViMTE4Mzg1YThiMjU5NTY4IiwiaCI6Im11cm11cjY0In0="
ORS_API_KEY = os.environ.get("ORS_API_KEY")
print("Blind Assist Server starting...")

OBSTACLES = [
    "person", "car", "truck", "bus", "motorcycle",
    "bicycle", "chair", "dining table", "couch", "bed",
    "toilet", "dog", "cat", "potted plant", "bottle",
    "suitcase"
]

LANDMARKS = [
    "door", "stairs", "traffic light",
    "stop sign", "bench", "fire hydrant"
]

def get_zone(box_center_x, frame_width):
    left_boundary  = frame_width * 0.33
    right_boundary = frame_width * 0.66
    if box_center_x < left_boundary:
        return "left"
    elif box_center_x > right_boundary:
        return "right"
    else:
        return "center"

def get_direction(zone):
    if zone == "center":
        return "stop"
    elif zone == "left":
        return "move right"
    else:
        return "move left"

def get_distance(box_height, frame_height):
    ratio = box_height / frame_height
    if ratio > 0.5:
        return "very_close"
    elif ratio > 0.3:
        return "close"
    elif ratio > 0.15:
        return "nearby"
    else:
        return "far"

def build_english_instruction(
        obstacles, landmarks, alert_level):
    if not obstacles and not landmarks:
        return "Path is clear."
    if obstacles:
        o = obstacles[0]
        if o["zone"] == "center":
            return (f"Stop! {o['name']} ahead, "
                    f"{o['distance']}.")
        elif o["zone"] == "left":
            return f"{o['name']} on left. Move right."
        else:
            return f"{o['name']} on right. Move left."
    return "Path is clear."

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Blind Assist server is alive!",
        "version": "2.0"
    })

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "Blind Assist Navigation Server running!"
    })

@app.route("/navigate", methods=["POST"])
def navigate():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(
            io.BytesIO(image_file.read())).convert("RGB")
        frame_width, frame_height = image.size

        yolo = get_model()
        results = yolo(image, verbose=False)
        result  = results[0]

        obstacles   = []
        landmarks   = []
        alert_level = "clear"

        for box in result.boxes:
            class_id   = int(box.cls)
            name       = result.names[class_id]
            confidence = float(box.conf)

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x   = (x1 + x2) / 2
            box_height = y2 - y1

            zone      = get_zone(center_x, frame_width)
            distance  = get_distance(
                box_height, frame_height)
            direction = get_direction(zone)

            if name in OBSTACLES:
                obstacles.append({
                    "name":      name,
                    "zone":      zone,
                    "distance":  distance,
                    "direction": direction
                })
                if (distance in ["very_close", "close"]
                        and zone == "center"):
                    alert_level = "danger"
                elif alert_level != "danger":
                    alert_level = "warning"

            elif name in LANDMARKS:
                landmarks.append({
                    "name": name,
                    "zone": zone
                })

        # Build structured keys for Flutter translation
        instruction_key = "path_clear"
        object_name     = ""
        distance_key    = ""

        if obstacles:
            danger_obs = [
                o for o in obstacles
                if o["zone"] == "center"
                and o["distance"] in [
                    "very_close", "close"]
            ]

            if danger_obs:
                instruction_key = "obstacle_ahead"
                object_name     = danger_obs[0]["name"]
                distance_key    = danger_obs[0]["distance"]
            else:
                center_obs = [o for o in obstacles
                    if o["zone"] == "center"]
                left_obs   = [o for o in obstacles
                    if o["zone"] == "left"]
                right_obs  = [o for o in obstacles
                    if o["zone"] == "right"]

                if center_obs:
                    instruction_key = "obstacle_ahead"
                    object_name     = center_obs[0]["name"]
                    distance_key    = center_obs[0]["distance"]
                elif left_obs:
                    instruction_key = "obstacle_left"
                    object_name     = left_obs[0]["name"]
                    distance_key    = left_obs[0]["distance"]
                elif right_obs:
                    instruction_key = "obstacle_right"
                    object_name     = right_obs[0]["name"]
                    distance_key    = right_obs[0]["distance"]

        return jsonify({
            "obstacles":         obstacles,
            "landmarks":         landmarks,
            "alert_level":       alert_level,
            "instruction_key":   instruction_key,
            "object_name":       object_name,
            "distance_key":      distance_key,
            "voice_instruction": build_english_instruction(
                obstacles, landmarks, alert_level)
        })

    except Exception as e:
        print(f"Navigate error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/directions", methods=["POST"])
def get_directions():
    try:
        data        = request.json
        origin_lat  = data.get("lat")
        origin_lng  = data.get("lng")
        destination = data.get("destination")

        geocode_url    = (
            "https://api.openrouteservice.org"
            "/geocode/search"
        )
        geocode_params = {
            "api_key": ORS_API_KEY,
            "text":    destination,
            "size":    1
        }

        geo_response = req.get(
            geocode_url,
            params=geocode_params,
            timeout=10
        )
        geo_data = geo_response.json()

        if not geo_data.get("features"):
            return jsonify({
                "error": "Destination not found",
                "tip": ("Try adding city name, "
                        "e.g. 'market, Bangarmau, UP'")
            }), 400

        dest_coords = (
            geo_data["features"][0]
            ["geometry"]["coordinates"]
        )
        dest_lng  = dest_coords[0]
        dest_lat  = dest_coords[1]
        dest_name = (
            geo_data["features"][0]
            ["properties"]["label"]
        )

        directions_url = (
            "https://api.openrouteservice.org"
            "/v2/directions/foot-walking"
        )
        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type":  "application/json"
        }
        body = {
            "coordinates": [
                [origin_lng, origin_lat],
                [dest_lng,   dest_lat]
            ],
            "instructions": True,
            "language":     "en"
        }

        dir_response = req.post(
            directions_url,
            json=body,
            headers=headers,
            timeout=10
        )
        dir_data = dir_response.json()

        if "routes" not in dir_data:
            return jsonify({
                "error":  "No route found",
                "detail": str(dir_data)
            }), 400

        route    = dir_data["routes"][0]
        summary  = route["summary"]
        segments = route["segments"][0]["steps"]

        steps = []
        for step in segments:
            steps.append({
                "instruction": step["instruction"],
                "distance": (
                    f"{round(step['distance'])} meters"),
                "duration": (
                    f"{round(step['duration'])} seconds"),
                "end_lat": dest_lat,
                "end_lng": dest_lng,
            })

        total_distance = (
            f"{round(summary['distance'])} meters")
        total_duration = (
            f"{round(summary['duration'] / 60)} minutes")

        return jsonify({
            "steps":             steps,
            "total_distance":    total_distance,
            "total_duration":    total_duration,
            "destination_name":  dest_name,
            "first_instruction": (
                steps[0]["instruction"]
                if steps else "Start walking")
        })

    except Exception as e:
        print(f"Directions error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)