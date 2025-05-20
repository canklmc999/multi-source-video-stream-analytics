import cv2
from detection import detect_objects

def get_gst_pipeline_gstreamer(device_name, width=640, height=480, fps=30):
    return (
        f'dshowvideosrc device-name="{device_name}" ! '
        f'video/x-raw, width={width}, height={height}, framerate={fps}/1 ! '
        f'videoconvert ! appsink'
    )

def main():
    cam1_name = "A4 TECH HD PC Camera"
    cam2_name = "HD Camera"

    pipe1 = get_gst_pipeline_gstreamer(cam1_name)
    pipe2 = get_gst_pipeline_gstreamer(cam2_name)

    print("Pipeline 1:", pipe1)
    print("Pipeline 2:", pipe2)

    cap1 = cv2.VideoCapture(pipe1, cv2.CAP_GSTREAMER)
    cap2 = cv2.VideoCapture(pipe2, cv2.CAP_GSTREAMER)

    if not cap1.isOpened():
        print(f"❌ Failed to open: {cam1_name}")
    if not cap2.isOpened():
        print(f"❌ Failed to open: {cam2_name}")
    if not cap1.isOpened() or not cap2.isOpened():
        return

    print("✅ Both cameras opened successfully with GStreamer.")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("❌ Failed to read from one or both cameras.")
            break

        # Apply YOLO detection and annotate frames
        frame1 = detect_objects(frame1)
        frame2 = detect_objects(frame2)

        cv2.imshow("A4 TECH HD PC Camera", frame1)
        cv2.imshow("HD Camera", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()