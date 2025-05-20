from flask import Flask, jsonify, Response
from threading import Thread
import cv2
from detection import detect_objects

app = Flask(__name__)

# Define camera device names
CAMERA_SOURCES = {
    "cam1": "A4 TECH HD PC Camera",
    "cam2": "HD Camera"
}

# Track threads and running states
camera_threads = {}
running_flags = {}

def get_gst_pipeline(device_name, width=640, height=480, fps=30):
    return (
        f'dshowvideosrc device-name="{device_name}" ! '
        f'video/x-raw, width={width}, height={height}, framerate={fps}/1 ! '
        f'videoconvert ! appsink'
    )

# Threaded capture loop for live display with detection
def camera_loop(cam_id, device_name):
    pipeline = get_gst_pipeline(device_name)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"❌ Failed to open: {cam_id}")
        return

    print(f"✅ {cam_id} started.")
    while running_flags.get(cam_id, False):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO object detection
        annotated = detect_objects(frame)
        cv2.imshow(cam_id, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running_flags[cam_id] = False
            break

    cap.release()
    cv2.destroyWindow(cam_id)
    print(f"⛔ {cam_id} stopped.")

# REST API: Start camera stream
@app.route("/start/<cam_id>", methods=["POST"])
def start_camera(cam_id):
    if cam_id in CAMERA_SOURCES and not running_flags.get(cam_id, False):
        running_flags[cam_id] = True
        thread = Thread(target=camera_loop, args=(cam_id, CAMERA_SOURCES[cam_id]))
        thread.start()
        camera_threads[cam_id] = thread
        return jsonify({"status": f"{cam_id} started"}), 200
    return jsonify({"error": "Invalid or already running"}), 400

# REST API: Stop camera stream
@app.route("/stop/<cam_id>", methods=["POST"])
def stop_camera(cam_id):
    if cam_id in running_flags and running_flags[cam_id]:
        running_flags[cam_id] = False
        return jsonify({"status": f"{cam_id} stopping"}), 200
    return jsonify({"error": "Camera not running"}), 400

# REST API: Check status of all cameras
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        cam_id: "running" if running_flags.get(cam_id, False) else "stopped"
        for cam_id in CAMERA_SOURCES
    })

# MJPEG stream generator (for client.py)
def generate_stream(cam_id):
    device_name = CAMERA_SOURCES[cam_id]
    pipeline = get_gst_pipeline(device_name)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        yield b''

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detect_objects(frame)
        _, jpeg = cv2.imencode('.jpg', annotated)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

# Endpoint for MJPEG stream
@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in CAMERA_SOURCES:
        return "Invalid camera", 404
    return Response(generate_stream(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
