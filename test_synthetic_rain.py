#!/usr/bin/env python3
"""
Synthetic Rain & Home Assistant Integration Test Script
======================================================
Tests rain detection algorithms and Home Assistant API publishing
using simulated video frames and a local mock HTTP server.
"""

import http.server
import json
import sys
import threading
import time
import urllib.request
import cv2
import numpy as np

from rain_detector import RainDetector, HomeAssistantPublisher

class MockHAServer(http.server.BaseHTTPRequestHandler):
    received_payloads = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body)
        auth = self.headers.get("Authorization", "")
        
        MockHAServer.received_payloads.append({
            "path": self.path,
            "auth": auth,
            "payload": payload,
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"result": "success"}')

    def log_message(self, format, *args):
        pass  # Suppress HTTP server output in test logs


class FrameGenerator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.drops = []
        for _ in range(60):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            speed = np.random.randint(15, 30)
            length = np.random.randint(15, 30)
            self.drops.append([x, y, speed, length])

    def get_frame(self, is_raining=True):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(0, self.height, 10):
            val = int(80 + (y / self.height) * 60)
            frame[y:y+10, :] = (val, val, val)

        if is_raining:
            for drop in self.drops:
                x, y, speed, length = drop
                cv2.line(frame, (x, y), (x, y + length), (240, 240, 240), 1)
                drop[1] += speed
                if drop[1] > self.height:
                    drop[1] = np.random.randint(-40, 0)
                    drop[0] = np.random.randint(0, self.width)
                    
        return frame


def run_test():
    print("==================================================")
    print(" Running Rain Detector & Home Assistant Test Suite ")
    print("==================================================\n")
    
    # Start Local Mock Home Assistant HTTP Server
    server = http.server.HTTPServer(("127.0.0.1", 18123), MockHAServer)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    ha_publisher = HomeAssistantPublisher(
        ha_url="http://127.0.0.1:18123",
        token="mock_test_token_12345",
        entity_id="binary_sensor.webcam_rain_detector",
        heartbeat_interval=1.0,
    )
    
    stream = FrameGenerator()
    detector = RainDetector(rain_threshold=8, heavy_rain_threshold=25)
    
    print("1. Testing CLEAR scenario (No rain)...")
    for i in range(30):
        frame = stream.get_frame(is_raining=False)
        res = detector.process_frame(frame)
        ha_publisher.update(res["is_raining"], res["rain_level"], res["avg_streaks"])
        time.sleep(0.005)
    
    print(f"   Clear test result: 30f avg streaks = {detector.avg_streaks:.1f}, status = {detector.rain_level}")
    assert not detector.is_raining, f"FAIL: False positive rain detected! ({detector.rain_level})"
    print("   [PASS] Clear scenario passed!\n")
    
    print("2. Testing RAIN scenario (Falling rain streaks)...")
    for i in range(40):
        frame = stream.get_frame(is_raining=True)
        res = detector.process_frame(frame)
        ha_publisher.update(res["is_raining"], res["rain_level"], res["avg_streaks"])
        time.sleep(0.005)
        
    print(f"   Rain test result: 30f avg streaks = {detector.avg_streaks:.1f}, status = {detector.rain_level}")
    assert detector.is_raining, f"FAIL: Failed to detect rain! ({detector.rain_level})"
    print(f"   [PASS] Rain scenario passed! Detected status: {detector.rain_level}\n")
    
    print("3. Verifying Home Assistant Sensor Payloads...")
    assert len(MockHAServer.received_payloads) >= 2, "FAIL: Home Assistant server received insufficient payloads!"
    
    # Verify CLEAR payload
    clear_payload = MockHAServer.received_payloads[0]
    print(f"   Received CLEAR HA Payload: {clear_payload['payload']}")
    assert clear_payload["payload"]["state"] == "off"
    assert clear_payload["payload"]["attributes"]["is_raining"] is False
    assert clear_payload["auth"] == "Bearer mock_test_token_12345"
    
    # Verify RAIN payload
    rain_payload = MockHAServer.received_payloads[-1]
    print(f"   Received RAIN HA Payload:  {rain_payload['payload']}")
    assert rain_payload["payload"]["state"] == "on"
    assert rain_payload["payload"]["attributes"]["is_raining"] is True
    print("   [PASS] Home Assistant API payloads verified!\n")

    server.shutdown()
    
    print("==================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY!                   ")
    print("==================================================")

if __name__ == "__main__":
    run_test()
