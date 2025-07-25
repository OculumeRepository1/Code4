import cv2
import threading

def gstreamer_pipeline(rtsp_url, latency=200):
    return (
        f"rtspsrc location={rtsp_url} latency={latency} ! "
        f"rtph264depay ! h264parse ! avdec_h264 ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )

class CameraThread(threading.Thread):
    def __init__(self, source, name="Camera"):
        super().__init__()
        self.source = source
        self.name = name
        self.running = True

        # Decide source type
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)  # Local webcam
        elif isinstance(source, str) and source.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(gstreamer_pipeline(source), cv2.CAP_GSTREAMER)
        else:
            raise ValueError(f"[{self.name}] Unsupported source: {source}")

    def run(self):
        if not self.cap.isOpened():
            print(f"[{self.name}] Could not open stream.")
            return
        
        print(f"[{self.name}] Streaming started.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.name}] Frame read failed.")
                break

            # Draw camera name
            cv2.putText(frame, self.name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the feed
            cv2.imshow(self.name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        self.cap.release()
        cv2.destroyWindow(self.name)
        print(f"[{self.name}] Streaming stopped.")

    def stop(self):
        self.running = False

def launch_multi_camera(sources):
    threads = []

    for i, src in enumerate(sources):
        cam_name = f"Camera-{i+1}"
        thread = CameraThread(source=src, name=cam_name)
        thread.start()
        threads.append(thread)

    try:
        while any(t.is_alive() for t in threads):
            pass
    except KeyboardInterrupt:
        print("Stopping all cameras...")
        for t in threads:
            t.stop()

sources = [
    0,  # Local webcam
    "rtsp://admin:Oculume2024@192.168.1.95/Preview_01_sub" 
    ]
launch_multi_camera(sources)