# RTSP Webcam Rain Detector with Home Assistant Integration

A high-performance Python script for real-time rainfall detection from RTSP camera streams, webcams, or video files using OpenCV, integrated directly with Home Assistant.

## Features

- **Home Assistant REST API Integration**: Dynamically updates a `binary_sensor.webcam_rain_detector` entity in Home Assistant (`on`/`true` when raining, `off`/`false` when clear).
- **Decoupled Threaded RTSP Reader**: Prevents RTSP stream buffer lag and latency build-up by reading frames in a background thread.
- **Automatic Connection Recovery**: Reconnects automatically if network glitches or camera stream drops occur.
- **Directional & Aspect Ratio Filtering**: Employs MOG2 background subtraction with custom vertical/diagonal morphological kernels to isolate rain streaks from general scene motion.
- **Temporal Sliding Window**: Uses a 30-frame rolling average to avoid false triggers caused by sudden lighting shifts or flying insects.
- **Telemetry HUD & Headless Support**: Displays a live visual overlay on-screen or runs silently in headless mode for server/edge deployments.

## Quick Start

### 1. Installation

Set up a virtual environment and install dependencies:

```bash
cd ~/Documents/rain_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Home Assistant Setup

1. Log into your Home Assistant instance.
2. Go to your **Profile** (click your username at the bottom of the left sidebar).
3. Scroll down to **Long-Lived Access Tokens** and click **Create Token**.
4. Name it `Webcam Rain Detector` and copy the generated token.

### 3. Usage Examples

#### Run with Home Assistant Integration via CLI Flags
```bash
python rain_detector.py \
  --rtsp "rtsp://admin:password@192.168.1.100:554/stream" \
  --ha-url "http://192.168.1.50:8123" \
  --ha-token "YOUR_LONG_LIVED_ACCESS_TOKEN" \
  --ha-entity "binary_sensor.webcam_rain_detector"
```

#### Run using Environment Variables (Recommended for Systemd / Docker)
```bash
export HA_URL="http://homeassistant.local:8123"
export HA_TOKEN="YOUR_LONG_LIVED_ACCESS_TOKEN"
export HA_ENTITY_ID="binary_sensor.webcam_rain_detector"

python rain_detector.py --rtsp "rtsp://camera_ip:554/stream" --headless
```

---

## Home Assistant Entity Details

The script automatically creates and manages a Home Assistant binary sensor:

- **Entity ID**: `binary_sensor.webcam_rain_detector` (or your custom entity ID)
- **State**:
  - `on`: Currently Raining (`is_raining: true`)
  - `off`: Clear Sky / Not Raining (`is_raining: false`)
- **Attributes**:
  ```json
  {
    "is_raining": true,
    "rain_level": "LIGHT RAIN",
    "average_streaks": 18.5,
    "device_class": "moisture",
    "friendly_name": "Webcam Rain Detector",
    "icon": "mdi:weather-pouring"
  }
  ```

---

## CLI Command Options

| Argument | Description | Default / Env Var |
| :--- | :--- | :--- |
| `--rtsp` | **(Required)** RTSP URL, video file path, or camera index | None |
| `--ha-url` | Home Assistant server base URL (e.g. `http://192.168.1.50:8123`) | `HA_URL` |
| `--ha-token` | Home Assistant Long-Lived Access Token | `HA_TOKEN` |
| `--ha-entity` | Home Assistant entity ID | `binary_sensor.webcam_rain_detector` / `HA_ENTITY_ID` |
| `--threshold` | Average streak count threshold for `LIGHT RAIN` | `12.0` |
| `--heavy-threshold` | Average streak count threshold for `HEAVY RAIN` | `35.0` |
| `--show` | Show live GUI window with HUD telemetry overlay | `True` |
| `--headless` | Run in headless mode without opening GUI window | `False` |

---

## Testing

Run the included test suite to verify the rain detector and Home Assistant API publisher:

```bash
python test_synthetic_rain.py
```
