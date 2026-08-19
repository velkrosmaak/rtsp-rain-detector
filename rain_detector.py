#!/usr/bin/env python3
"""
RTSP Webcam Rain Detector with Home Assistant Integration
==========================================================
Detects rainfall in real-time from an RTSP camera stream (or video file / webcam)
using OpenCV computer vision techniques (Background Subtraction + Morphological Streak Analysis)
and updates a Home Assistant binary sensor (`is_raining`: true / false).

Features:
- Threaded RTSP Stream Reader: Prevents buffer lag/latency accumulation.
- Automatic Reconnection: Retries connection if stream drops.
- Directional Streak Filtering: Filters moving particles by aspect ratio, orientation, and size.
- Rolling Temporal Analysis: Prevents false positives using sliding window confidence scoring.
- Home Assistant Integration: Publishes `true`/`false` rain state to `binary_sensor.webcam_rain_detector` via REST API.
- Visual Overlay & Headless Modes: Displays HUD telemetry or runs silently as a background service.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from threading import Lock, Thread

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RainDetector")


class RTSPStreamReader:
    """
    Threaded video capture class for RTSP streams.
    Decouples frame grabbing from frame processing to eliminate RTSP buffering latency.
    """

    def __init__(self, src, reconnect_delay=5):
        """
        :param src: RTSP URL (str), video file path (str), or camera index (int)
        :param reconnect_delay: Seconds to wait before attempting reconnection
        """
        # Convert numeric string to int if camera index was provided
        if isinstance(src, str) and src.isdigit():
            self.src = int(src)
        else:
            self.src = src

        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = Lock()
        self.thread = None
        self.fps = 0.0

    def start(self):
        """Start the background frame grabbing thread."""
        self._connect()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _connect(self):
        """Attempt connection to the video source."""
        logger.info(f"Connecting to video source: {self.src}")
        if self.cap is not None:
            self.cap.release()
        
        # Enable FFmpeg TCP transport if RTSP for better stability over UDP
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            logger.warning(f"Unable to open source {self.src}")
            self.ret = False
        else:
            logger.info("Successfully connected to video stream.")
            self.ret, self.frame = self.cap.read()

    def _update(self):
        """Continuously grab frames from the stream."""
        last_time = time.time()
        frame_count = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logger.warning("Stream connection lost. Reconnecting...")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            grabbed, frame = self.cap.read()
            if not grabbed:
                logger.warning("Failed to grab frame. Retrying...")
                time.sleep(0.1)
                if not self.cap.isOpened():
                    self._connect()
                continue

            # Update FPS calculation
            frame_count += 1
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                last_time = now

            with self.lock:
                self.ret = grabbed
                self.frame = frame

    def read(self):
        """Read the most recent frame thread-safely."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Stop the background thread and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("RTSP reader stopped.")


class RainDetector:
    """
    Computer Vision Rain Detector using MOG2 Background Subtraction,
    Morphological Streak Kernels, Contour Geometry Filtering, and Temporal Smoothing.
    """

    def __init__(
        self,
        min_streak_area=8,
        max_streak_area=800,
        min_aspect_ratio=2.2,
        max_angle_deg=45.0,
        rain_threshold=12,
        heavy_rain_threshold=35,
        history_window=30,
    ):
        """
        :param min_streak_area: Minimum area in pixels for a rain streak blob
        :param max_streak_area: Maximum area in pixels for a rain streak blob
        :param min_aspect_ratio: Minimum height/width or length/width ratio for streaks
        :param max_angle_deg: Max deviation from vertical axis (90°) allowed for rain streaks
        :param rain_threshold: Rolling average streak count to trigger 'LIGHT RAIN'
        :param heavy_rain_threshold: Rolling average streak count to trigger 'HEAVY RAIN'
        :param history_window: Number of frames in sliding window for temporal smoothing
        """
        self.min_area = min_streak_area
        self.max_area = max_streak_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_angle_deg = max_angle_deg
        self.rain_threshold = rain_threshold
        self.heavy_rain_threshold = heavy_rain_threshold

        # Background subtractor tuned for dynamic fast-moving foreground objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False
        )

        # Directional kernel to emphasize vertical/slanted streaks
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        # Temporal rolling history for streak count smoothing
        self.streak_history = deque(maxlen=history_window)
        
        # State tracking
        self.is_raining = False
        self.rain_level = "CLEAR"  # 'CLEAR', 'LIGHT RAIN', 'HEAVY RAIN'
        self.avg_streaks = 0.0

    def process_frame(self, frame):
        """
        Process a single image frame and detect rain streaks.

        :param frame: BGR numpy image frame
        :return: dict containing detection results, streak bounding boxes, and processed mask
        """
        if frame is None:
            return None

        # 1. Preprocessing: convert to Grayscale and apply subtle blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 2. Background Subtraction to capture dynamic foreground elements
        fg_mask = self.bg_subtractor.apply(blurred)

        # Remove static/noise elements with morphological operations
        enhanced_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.vertical_kernel)
        _, binary_mask = cv2.threshold(enhanced_mask, 200, 255, cv2.THRESH_BINARY)

        # 3. Contour Detection & Geometric Filtering
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_streaks = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            aspect_ratio = float(h) / float(w)

            angle_valid = False
            if len(cnt) >= 5:
                (ellipse_center, (d1, d2), angle) = cv2.fitEllipse(cnt)
                vert_angle = abs(angle - 90.0) if angle <= 90 else abs(angle - 90.0)
                if vert_angle <= self.max_angle_deg:
                    angle_valid = True
                    major_axis = max(d1, d2)
                    minor_axis = min(d1, d2)
                    if minor_axis > 0:
                        aspect_ratio = max(aspect_ratio, major_axis / minor_axis)
            else:
                if aspect_ratio >= self.min_aspect_ratio:
                    angle_valid = True

            if aspect_ratio >= self.min_aspect_ratio and angle_valid:
                detected_streaks.append((x, y, w, h))

        # 4. Temporal Smoothing
        current_streak_count = len(detected_streaks)
        self.streak_history.append(current_streak_count)
        self.avg_streaks = sum(self.streak_history) / len(self.streak_history)

        # Update rain detection status
        if self.avg_streaks >= self.heavy_rain_threshold:
            self.is_raining = True
            self.rain_level = "HEAVY RAIN"
        elif self.avg_streaks >= self.rain_threshold:
            self.is_raining = True
            self.rain_level = "LIGHT RAIN"
        else:
            self.is_raining = False
            self.rain_level = "CLEAR"

        return {
            "is_raining": self.is_raining,
            "rain_level": self.rain_level,
            "current_streaks": current_streak_count,
            "avg_streaks": round(self.avg_streaks, 1),
            "streak_boxes": detected_streaks,
            "mask": binary_mask,
        }

    def draw_hud(self, frame, result, fps=0.0):
        """
        Draw visual overlay (HUD) on top of the frame.
        """
        if frame is None or result is None:
            return frame

        annotated = frame.copy()

        # Highlight detected rain streaks in cyan
        for x, y, w, h in result["streak_boxes"]:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 1)

        # Determine HUD colors based on rain state
        if result["rain_level"] == "HEAVY RAIN":
            status_color = (0, 0, 255)  # Red
        elif result["rain_level"] == "LIGHT RAIN":
            status_color = (0, 215, 255)  # Amber/Yellow
        else:
            status_color = (0, 255, 0)  # Green

        # Draw Telemetry HUD Panel
        panel_h, panel_w = 110, 320
        cv2.rectangle(annotated, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (10 + panel_w, 10 + panel_h), status_color, 2)

        # Text strings
        status_text = f"STATUS: {result['rain_level']}"
        streaks_text = f"Streaks (instant): {result['current_streaks']}"
        avg_text = f"Streaks (30f avg): {result['avg_streaks']:.1f} / {self.rain_threshold}"
        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(
            annotated,
            status_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            streaks_text,
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            avg_text,
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            fps_text,
            (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return annotated


class HomeAssistantPublisher:
    """
    Publishes rain detection status to a Home Assistant binary sensor via REST API.
    State is 'on' when raining (is_raining: true) and 'off' when not (is_raining: false).
    """

    def __init__(
        self,
        ha_url,
        token,
        entity_id="binary_sensor.webcam_rain_detector",
        heartbeat_interval=30.0,
    ):
        """
        :param ha_url: Home Assistant base URL (e.g. 'http://192.168.1.50:8123')
        :param token: Long-lived Access Token
        :param entity_id: Entity ID in Home Assistant (e.g. 'binary_sensor.webcam_rain_detector')
        :param heartbeat_interval: Seconds between periodic state refresh pushes
        """
        self.ha_url = ha_url.rstrip("/")
        self.token = token
        
        # Ensure entity starts with binary_sensor. or sensor.
        if not (entity_id.startswith("binary_sensor.") or entity_id.startswith("sensor.")):
            self.entity_id = f"binary_sensor.{entity_id}"
        else:
            self.entity_id = entity_id

        self.heartbeat_interval = heartbeat_interval
        self.api_endpoint = f"{self.ha_url}/api/states/{self.entity_id}"
        self.last_published_state = None
        self.last_publish_time = 0.0

    def update(self, is_raining, rain_level, avg_streaks, force=False):
        """
        Publish updated rain state to Home Assistant.
        Triggered immediately on state change, or periodically based on heartbeat_interval.
        """
        now = time.time()
        state_changed = (is_raining != self.last_published_state)
        time_elapsed = (now - self.last_publish_time) >= self.heartbeat_interval

        if not force and not state_changed and not time_elapsed:
            return

        # Home Assistant binary_sensor uses 'on' / 'off' state
        state_str = "on" if is_raining else "off"

        payload = {
            "state": state_str,
            "attributes": {
                "is_raining": bool(is_raining),
                "rain_level": rain_level,
                "average_streaks": float(avg_streaks),
                "device_class": "moisture",
                "friendly_name": "Driveway Camera Rain Detector",
                "icon": "mdi:weather-pouring" if is_raining else "mdi:weather-sunny",
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status in (200, 201):
                    self.last_published_state = is_raining
                    self.last_publish_time = now
                    logger.info(
                        f"Home Assistant sensor updated -> {self.entity_id} = {state_str} (is_raining: {is_raining})"
                    )
                else:
                    logger.warning(f"Home Assistant returned HTTP status code {response.status}")
        except urllib.error.URLError as e:
            logger.error(f"Failed to reach Home Assistant endpoint ({self.api_endpoint}): {e}")
        except Exception as e:
            logger.error(f"Error publishing to Home Assistant: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect rain in real-time from an RTSP webcam stream and update Home Assistant."
    )
    parser.add_argument(
        "--rtsp",
        type=str,
        required=True,
        help="RTSP URL (e.g., rtsp://user:pass@190.168.1.50:554/stream), video file path, or camera index (0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=12.0,
        help="Streak threshold for Light Rain detection (default: 12.0).",
    )
    parser.add_argument(
        "--heavy-threshold",
        type=float,
        default=35.0,
        help="Streak threshold for Heavy Rain detection (default: 35.0).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display live GUI window with visual annotations.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI window and run in CLI headless mode.",
    )
    
    # Home Assistant Options (Fallback to environment variables if set)
    parser.add_argument(
        "--ha-url",
        type=str,
        default=os.getenv("HA_URL", ""),
        help="Home Assistant URL (e.g. http://192.168.1.50:8123 or http://homeassistant.local:8123). Can also use HA_URL env var.",
    )
    parser.add_argument(
        "--ha-token",
        type=str,
        default=os.getenv("HA_TOKEN", ""),
        help="Home Assistant Long-Lived Access Token. Can also use HA_TOKEN env var.",
    )
    parser.add_argument(
        "--ha-entity",
        type=str,
        default=os.getenv("HA_ENTITY_ID", "binary_sensor.webcam_rain_detector"),
        help="Home Assistant entity ID (default: binary_sensor.webcam_rain_detector).",
    )
    
    args = parser.parse_args()

    show_gui = args.show and not args.headless

    logger.info("Initializing RTSP Rain Detector...")
    reader = RTSPStreamReader(args.rtsp).start()
    detector = RainDetector(
        rain_threshold=args.threshold,
        heavy_rain_threshold=args.heavy_threshold,
    )

    # Initialize Home Assistant publisher if configured
    ha_publisher = None
    if args.ha_url and args.ha_token:
        logger.info(f"Configuring Home Assistant integration for entity: {args.ha_entity}")
        ha_publisher = HomeAssistantPublisher(
            ha_url=args.ha_url,
            token=args.ha_token,
            entity_id=args.ha_entity,
        )
    else:
        logger.info("Home Assistant integration disabled. Provide --ha-url and --ha-token to enable.")

    last_state = None

    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            result = detector.process_frame(frame)

            # Log status updates on state change
            if result["rain_level"] != last_state:
                logger.info(
                    f"Rain state changed -> {result['rain_level']} (is_raining: {result['is_raining']}, 30f avg streaks: {result['avg_streaks']})"
                )
                last_state = result["rain_level"]

            # Update Home Assistant sensor
            if ha_publisher:
                ha_publisher.update(
                    is_raining=result["is_raining"],
                    rain_level=result["rain_level"],
                    avg_streaks=result["avg_streaks"],
                )

            if show_gui:
                hud_frame = detector.draw_hud(frame, result, fps=reader.fps)
                cv2.imshow("RTSP Rain Detector", hud_frame)

                # Press 'q' to quit, 'm' to show mask window
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User quit program.")
                    break
                elif key == ord("m"):
                    cv2.imshow("Foreground Streak Mask", result["mask"])
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        reader.stop()
        if show_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
