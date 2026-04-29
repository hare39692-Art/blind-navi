from flask import Flask, request, jsonify, session
from flask_cors import CORS
from PIL import Image
import io
import re
import requests as req
import os
from collections import defaultdict
import time
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "blind-assist-secret")

# Load YOLO lazily — only when first request comes in
model = None

# Room mapping and session tracking
room_layout = {}  # Store detected room layout per session
session_obstacles = defaultdict(list)  # Track obstacles over time
camera_status = {"working": False, "last_check": None, "error": None}  # Track camera status

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

def find_exits(landmarks):
    """Find and prioritize exits (doors, stairs) for blind user."""
    exits = [l for l in landmarks if l["name"] in ["door", "stairs"]]
    if not exits:
        return None
    # Prioritize exits in center, then left, then right
    zone_priority = {"center": 0, "left": 1, "right": 2}
    exits.sort(key=lambda x: zone_priority.get(x["zone"], 3))
    return exits[0] if exits else None

def build_room_map(obstacles):
    """Create a simple room layout map from detected obstacles."""
    zones = {"left": [], "center": [], "right": []}
    for obs in obstacles:
        zones[obs["zone"]].append(obs["name"])
    return zones

def get_avoidance_strategy(obstacles, landmarks, frame_width):
    """Build detailed avoidance strategy for blind user."""
    if not obstacles and not landmarks:
        return "Path is clear. You can move forward safely."
    
    if obstacles:
        center_obs = [o for o in obstacles if o["zone"] == "center"]
        left_obs = [o for o in obstacles if o["zone"] == "left"]
        right_obs = [o for o in obstacles if o["zone"] == "right"]
        
        # Priority: detect very close obstacles first
        danger_obs = [o for o in obstacles 
                      if o["distance"] in ["very_close", "close"]]
        
        if danger_obs:
            obs = danger_obs[0]
            if obs["zone"] == "center":
                if left_obs:
                    return f"DANGER! {obs['name']} directly ahead. Recommend moving RIGHT to clear it."
                elif right_obs:
                    return f"DANGER! {obs['name']} directly ahead. Recommend moving LEFT to clear it."
                else:
                    return f"DANGER! {obs['name']} directly ahead. Move LEFT or RIGHT to avoid."
            elif obs["zone"] == "left":
                return f"{obs['name']} detected on LEFT, very close. Move RIGHT carefully and forward."
            else:
                return f"{obs['name']} detected on RIGHT, very close. Move LEFT carefully and forward."
        
        # Handle nearby obstacles
        if center_obs:
            obs = center_obs[0]
            if obs["distance"] == "nearby":
                return f"{obs['name']} ahead at {obs['distance']} distance. Prepare to move around it."
            return f"{obs['name']} ahead. Move to the side when closer."
        elif left_obs and right_obs:
            return f"Obstacles detected on both sides ({left_obs[0]['name']}, {right_obs[0]['name']}). Move forward carefully."
        elif left_obs:
            return f"{left_obs[0]['name']} on your LEFT. Safe to move forward or move RIGHT."
        elif right_obs:
            return f"{right_obs[0]['name']} on your RIGHT. Safe to move forward or move LEFT."
    
    # Suggest landmarks/exits
    exits = find_exits(landmarks)
    if exits:
        if exits["zone"] == "center":
            return f"EXIT detected ahead. Move forward towards the {exits['name']}."
        elif exits["zone"] == "left":
            return f"EXIT detected on your LEFT. Move LEFT to find the {exits['name']}."
        else:
            return f"EXIT detected on your RIGHT. Move RIGHT to find the {exits['name']}."
    
    return "Path is clear. Move forward carefully."

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

def get_fallback_directions(origin_lat, origin_lng):
    """Provide fallback indoor navigation when API fails."""
    return {
        "steps": [
            {
                "instruction": "Stand up and face forward. Camera will detect obstacles ahead.",
                "distance": "Initial",
                "duration": "Ready"
            },
            {
                "instruction": "Follow the voice guidance from the camera to navigate around obstacles.",
                "distance": "As you move",
                "duration": "Continuous"
            },
            {
                "instruction": "Listen for alerts about obstacles on your left, right, or center.",
                "distance": "Real-time",
                "duration": "Always active"
            }
        ],
        "total_distance": "Local navigation",
        "total_duration": "Guide until destination",
        "destination_name": "Current Location - Indoor Navigation",
        "first_instruction": "Stand up. Face forward. Camera will guide you.",
        "fallback_mode": True,
        "mode_message": "Map service unavailable. Using local obstacle detection mode."
    }

def check_camera_health():
    """Check if camera and OpenCV are working."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {
                "camera_status": "not_connected",
                "working": False,
                "message": "Camera not connected. Please check USB connection.",
                "action": "Plug in your camera device."
            }
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return {
                "camera_status": "not_responding",
                "working": False,
                "message": "Camera not responding. Device may be blocked.",
                "action": "Try restarting your device."
            }
        
        camera_status["working"] = True
        camera_status["last_check"] = time.time()
        camera_status["error"] = None
        
        return {
            "camera_status": "working",
            "working": True,
            "message": "Camera connected and ready. OpenCV is working.",
            "action": "You can start navigating now."
        }
    except Exception as e:
        error_msg = str(e)
        camera_status["working"] = False
        camera_status["error"] = error_msg
        return {
            "camera_status": "error",
            "working": False,
            "message": f"Camera error: {error_msg}",
            "action": "Try reconnecting the camera or restarting the app."
        }

def generate_continuous_guidance(obstacles, landmarks, alert_level):
    """Generate continuous, repeating guidance for blind user."""
    base_guidance = build_english_instruction(obstacles, landmarks, alert_level)
    avoidance = get_avoidance_strategy(obstacles, landmarks, 640)  # Default width
    
    if not obstacles and not landmarks:
        return {
            "guidance": "All clear. Path is safe. Move forward.",
            "keep_repeating": False,
            "repeat_interval": 2,
            "obstacle_count": 0,
            "landmark_count": 0
        }
    
    if obstacles:
        return {
            "guidance": avoidance,
            "keep_repeating": True,
            "repeat_interval": 1,  # Repeat every 1 second for dangers
            "obstacle_count": len(obstacles),
            "landmark_count": len(landmarks),
            "obstacles_detected": [o["name"] for o in obstacles],
            "alert_level": alert_level
        }
    
    return {
        "guidance": avoidance,
        "keep_repeating": True,
        "repeat_interval": 2,
        "obstacle_count": 0,
        "landmark_count": len(landmarks),
        "landmarks_detected": [l["name"] for l in landmarks]
    }

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

@app.route("/camera-check", methods=["GET"])
def camera_check():
    """Check if camera and OpenCV are working. Tell blind user status."""
    health = check_camera_health()
    return jsonify(health)

@app.route("/camera-status", methods=["GET"])
def camera_status_endpoint():
    """Get current camera status for blind user."""
    if camera_status["working"]:
        message = "Camera is WORKING. OpenCV is detecting objects. Ready to guide you."
    elif camera_status["error"]:
        message = f"Camera NOT working. Error: {camera_status['error']}. Please fix it."
    else:
        message = "Camera status UNKNOWN. Please check your connection."
    
    return jsonify({
        "camera_working": camera_status["working"],
        "status_message": message,
        "last_check": camera_status["last_check"],
        "error": camera_status["error"]
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

        # Update camera status
        camera_status["working"] = True
        camera_status["last_check"] = time.time()

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

        # Build room layout map
        room_map = build_room_map(obstacles)
        
        # Find exits for emergency guidance
        exit_found = find_exits(landmarks)
        exit_guidance = ""
        if exit_found:
            if exit_found["zone"] == "center":
                exit_guidance = f"Exit ({exit_found['name']}) detected ahead."
            elif exit_found["zone"] == "left":
                exit_guidance = f"Exit ({exit_found['name']}) on your left."
            else:
                exit_guidance = f"Exit ({exit_found['name']}) on your right."
        
        # Get enhanced avoidance strategy
        avoidance_strategy = get_avoidance_strategy(obstacles, landmarks, frame_width)
        
        # Generate continuous guidance
        continuous_guidance = generate_continuous_guidance(obstacles, landmarks, alert_level)
        
        # Build obstacle detection status message
        obstacles_status = ""
        if obstacles:
            obstacle_names = [o["name"] for o in obstacles]
            obstacles_status = f"OBSTACLES DETECTED: {', '.join(obstacle_names)}. Listen carefully!"
        else:
            obstacles_status = "No obstacles detected. Path appears clear."

        return jsonify({
            "obstacles":             obstacles,
            "landmarks":             landmarks,
            "alert_level":           alert_level,
            "instruction_key":       instruction_key,
            "object_name":           object_name,
            "distance_key":          distance_key,
            "voice_instruction":     build_english_instruction(
                obstacles, landmarks, alert_level),
            "avoidance_strategy":    avoidance_strategy,
            "room_layout":           room_map,
            "exit_guidance":         exit_guidance,
            "has_exit":              exit_found is not None,
            "detailed_guidance":     avoidance_strategy,
            # NEW: Camera and continuous guidance fields
            "camera_status": "working",
            "camera_message": "Camera is working. OpenCV is detecting objects.",
            "obstacles_status": obstacles_status,
            "obstacle_count": len(obstacles),
            "landmark_count": len(landmarks),
            "continuous_guidance": continuous_guidance,
            "keep_repeating": continuous_guidance["keep_repeating"],
            "repeat_interval_seconds": continuous_guidance["repeat_interval"]
        })

    except Exception as e:
        print(f"Navigate error: {e}")
        camera_status["working"] = False
        camera_status["error"] = str(e)
        return jsonify({
            "error": str(e),
            "camera_status": "error",
            "camera_message": f"Camera error: {str(e)}. Cannot detect obstacles."
        }), 500

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
            print("Destination not found - falling back to local navigation")
            return jsonify(get_fallback_directions(origin_lat, origin_lng)), 400

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
            print(f"No route found - falling back to local navigation: {dir_data}")
            return jsonify(get_fallback_directions(origin_lat, origin_lng)), 400

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
                if steps else "Start walking"),
            "fallback_mode":     False,
            "mode_message":      "GPS navigation active"
        })

    except Exception as e:
        print(f"Directions error: {e}")
        # Return fallback guidance instead of just error
        try:
            data = request.json
            origin_lat = data.get("lat")
            origin_lng = data.get("lng")
            return jsonify(get_fallback_directions(origin_lat, origin_lng))
        except:
            return jsonify({"error": str(e), "fallback_mode": True}), 500

@app.route("/find-exit", methods=["POST"])
def find_exit():
    """Emergency endpoint: Help blind user find nearest exit from current room."""
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
        exits_found = []

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
            distance  = get_distance(box_height, frame_height)

            if name in OBSTACLES:
                obstacles.append({
                    "name":     name,
                    "zone":     zone,
                    "distance": distance
                })
            elif name in LANDMARKS:
                landmarks.append({
                    "name": name,
                    "zone": zone
                })
                if name in ["door", "stairs"]:
                    exits_found.append({
                        "name": name,
                        "zone": zone
                    })

        # Find nearest exit
        exit_path = find_exits(landmarks)
        
        if not exit_path:
            return jsonify({
                "exit_found": False,
                "guidance": "No exit detected. Scan the room. Look for doors or stairs.",
                "instruction": "Turn left or right and scan again.",
                "obstacles_around": obstacles
            })

        exit_direction = ""
        if exit_path["zone"] == "center":
            exit_direction = "Go forward towards the exit."
            if obstacles:
                center_obs = [o for o in obstacles if o["zone"] == "center"]
                if center_obs:
                    exit_direction = f"Exit ahead but {center_obs[0]['name']} detected. Go around it carefully, then head to the exit."
        elif exit_path["zone"] == "left":
            exit_direction = "Turn left. Move towards the exit on your left."
        else:
            exit_direction = "Turn right. Move towards the exit on your right."

        return jsonify({
            "exit_found": True,
            "exit_type": exit_path["name"],
            "exit_zone": exit_path["zone"],
            "guidance": exit_direction,
            "instruction": f"Exit ({exit_path['name']}) detected. Follow this guidance to reach it.",
            "obstacles_in_path": [o for o in obstacles if o["zone"] == exit_path["zone"]],
            "all_obstacles": obstacles,
            "all_landmarks": landmarks
        })

    except Exception as e:
        print(f"Find exit error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/monitor", methods=["POST"])
def monitor():
    """
    Continuous monitoring endpoint - Call repeatedly to get updated guidance.
    Tells blind user if camera is working and what obstacles are detected.
    """
    if "image" not in request.files:
        return jsonify({
            "error": "No image sent",
            "camera_status": "error",
            "message": "Camera not responding. Check connection."
        }), 400

    try:
        image_file = request.files["image"]
        image = Image.open(
            io.BytesIO(image_file.read())).convert("RGB")
        frame_width, frame_height = image.size

        # Update camera status
        camera_status["working"] = True
        camera_status["last_check"] = time.time()
        camera_status["error"] = None

        yolo = get_model()
        results = yolo(image, verbose=False)
        result  = results[0]

        obstacles   = []
        landmarks   = []

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
            distance  = get_distance(box_height, frame_height)

            if name in OBSTACLES:
                obstacles.append({
                    "name":     name,
                    "zone":     zone,
                    "distance": distance
                })
            elif name in LANDMARKS:
                landmarks.append({
                    "name": name,
                    "zone": zone
                })

        # Determine current status
        if not obstacles and not landmarks:
            status_message = "Camera working. No obstacles detected. You can move forward safely."
            continuous = True
        elif obstacles:
            obstacle_list = [o["name"] for o in obstacles]
            status_message = f"CAMERA WORKING! OBSTACLES FOUND: {', '.join(obstacle_list)}. Listen for guidance."
            continuous = True
        else:
            landmark_list = [l["name"] for l in landmarks]
            status_message = f"Camera working. Landmarks detected: {', '.join(landmark_list)}."
            continuous = True

        guidance = generate_continuous_guidance(obstacles, landmarks, 
                                               "danger" if any(o["distance"] in ["very_close", "close"] 
                                                             and o["zone"] == "center" 
                                                             for o in obstacles) else "warning" if obstacles else "clear")

        return jsonify({
            "camera_status": "working",
            "camera_message": "OpenCV is actively detecting objects.",
            "status_message": status_message,
            "obstacles_detected": len(obstacles) > 0,
            "obstacle_count": len(obstacles),
            "obstacles": obstacles,
            "landmarks": landmarks,
            "guidance": guidance["guidance"],
            "keep_repeating": continuous,
            "repeat_interval_seconds": guidance["repeat_interval"],
            "timestamp": time.time()
        })

    except Exception as e:
        print(f"Monitor error: {e}")
        camera_status["working"] = False
        camera_status["error"] = str(e)
        return jsonify({
            "camera_status": "error",
            "camera_message": f"CAMERA ERROR: {str(e)}",
            "status_message": "Camera is NOT working. Cannot detect obstacles. Check your device connection.",
            "error": str(e)
        }), 500
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)