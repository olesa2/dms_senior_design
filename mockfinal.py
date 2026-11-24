"""
Alert logic:
  - Driver is "away" if the face is lost, OR head yaw/pitch is outside the tolerance
    (+/-35 deg yaw, 0 to +30 deg pitch).
  - Once "away" lasts longer than DISTRACTION_THRESHOLD_S, the alert fires:
        face lost            -> NO_FACE alert
        head turned/tilted   -> DISTRACTED alert
  - Alerts clear as soon as the driver is back within tolerance for at least 0.5s.
  - No drowsy alert for now (iris data to be integrarted).

WIRING  (BCM GPIO numbers; LCD uses 4-bit mode)
  LCD pin 1  VSS  -> GND
  LCD pin 2  VDD  -> 5V
  LCD pin 3  V0   -> GNF thru 1k resistor for fixed contrast.
  LCD pin 4  RS   -> GPIO25
  LCD pin 5  RW   -> GND   (write-only)
  LCD pin 6  E    -> GPIO24
  LCD pin 11 D4   -> GPIO23
  LCD pin 12 D5   -> GPIO17
  LCD pin 13 D6   -> GPIO18
  LCD pin 14 D7   -> GPIO22
  LCD pin 15 LED+ -> 5V    (backlight)
  LCD pin 16 LED- -> GND
  (LCD pins 7-10 are unused)

  Buzzer:  GPIO27 -> 1k resistor -> PN2222 base
           transistor emitter -> GND
           transistor collector -> buzzer(-)
           buzzer(+) -> 5V
  LED:     GPIO5 -> 330 ohm resistor -> LED anode;



"""

import subprocess
import threading
import time

import cv2
import numpy as np
import mediapipe as mp

import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd
from gpiozero import Buzzer, LED

# =====================================================================
# CONFIGURATION
# =====================================================================
# Image conditioning: "RAW", "HIST_EQ", or "CLAHE"
CONDITIONING_MODE = "CLAHE"
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2  # YUV420p bytes per frame

# Distraction logic
DISTRACTION_THRESHOLD_S = 3.0   # seconds "away" before alerting (3-5s)
YAW_LIMIT_DEG = 35.0  
PITCH_MIN_DEG = 0.0
PITCH_MAX_DEG = 30          # tolerance
REPEAT_BEEP_S = 4.0             # while still distracted, beep again this often

RUN_HAAR = False                 #Haar for comparison;

# LCD / buzzer / LED pins
LCD_RS = board.D25
LCD_EN = board.D24
LCD_D4 = board.D23
LCD_D5 = board.D17
LCD_D6 = board.D18
LCD_D7 = board.D22
LCD_COLS = 16
LCD_ROWS = 2
BUZZER_PIN = 27   # BCM, goes to transistor thru 1k
LED_PIN = 5       # BCM

# Beep patterns: (on_seconds, off_seconds, repeat_count)
PATTERNS = {
    "NO_FACE":    (0.10, 0.40, 2),
    "DISTRACTED": (0.20, 0.20, 3), 
}


# =====================================================================
# ALERT SYSTEM (LCD + buzzer + LED)
# =====================================================================
class AlertSystem:
    def __init__(self):
        def pin(p):
            return digitalio.DigitalInOut(p)

        self.lcd = characterlcd.Character_LCD_Mono(
            pin(LCD_RS), pin(LCD_EN),
            pin(LCD_D4), pin(LCD_D5), pin(LCD_D6), pin(LCD_D7),
            LCD_COLS, LCD_ROWS,
        )
        self.lcd.clear()
        self.buzzer = Buzzer(BUZZER_PIN)
        self.led = LED(LED_PIN)
        self._thread = None
        self._stop = threading.Event()
        self._last = None

    def _write(self, line1, line2=""):
        line1 = str(line1)[:LCD_COLS].ljust(LCD_COLS)
        line2 = str(line2)[:LCD_COLS].ljust(LCD_COLS)
        if (line1, line2) == self._last:
            return
        self._last = (line1, line2)
        self.lcd.message = line1 + "\n" + line2

    def show_status(self, line1, line2=""):
        """Normal (non-alert) status text."""
        self._write(line1, line2)

    def _beep(self, on, off, count):
        for _ in range(count):
            if self._stop.is_set():
                break
            self.buzzer.on()
            time.sleep(on)
            self.buzzer.off()
            time.sleep(off)
        self.buzzer.off()

    def alert(self, level, message=""):
        """LCD message + LED on + beeps in a background thread (non-blocking)."""
        self._write("!! " + level, message)
        self.led.on()
        if self._thread and self._thread.is_alive():
            return  # already beeping; don't stack beeps
        on, off, count = PATTERNS.get(level, (0.2, 0.2, 2))
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._beep, args=(on, off, count), daemon=True
        )
        self._thread.start()

    def clear_alert(self, line1="Driver OK", line2="Monitoring"):
        self._stop.set()
        self.buzzer.off()
        self.led.off()
        self._write(line1, line2)

    def close(self):
        self._stop.set()
        self.buzzer.off()
        self.led.off()
        self.lcd.clear()
        self.buzzer.close()
        self.led.close()


# =====================================================================
# CAMERA + VISION SETUP
# =====================================================================
alerts = AlertSystem()
alerts.show_status("DMS starting", "Please wait...")

cam_cmd = [
    "rpicam-vid", "-t", "0", "--nopreview",
    "--width", str(WIDTH), "--height", str(HEIGHT),
    "--framerate", "15", "--codec", "yuv420", "-o", "-"
]
cam_proc = subprocess.Popen(
    cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=FRAME_SIZE * 2
)

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  #enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#generic 3D face model
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],             # Nose tip (1)
    [0.0, -330.0, -65.0],        # Chin (152)
    [-225.0, 170.0, -135.0],     # Left eye left corner (263)
    [225.0, 170.0, -135.0],      # Right eye right corner (33)
    [-150.0, -150.0, -125.0],    # Left mouth corner (287)
    [150.0, -150.0, -125.0]      # Right mouth corner (57)
], dtype=np.float64)

LANDMARK_INDICES_POSE = [1, 152, 263, 33, 287, 57]


def capture_pi_frame():
    """Reads a single raw frame from rpicam-vid and converts to RGB."""
    raw_bytes = cam_proc.stdout.read(FRAME_SIZE)
    if len(raw_bytes) < FRAME_SIZE:
        return None
    yuv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)


def apply_conditioning(frame, mode):
    """Applies selected preprocessing filter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if mode == "HIST_EQ":
        cond_gray = cv2.equalizeHist(gray)
    elif mode == "CLAHE":
        cond_gray = clahe.apply(gray)
    else:  # "RAW"
        cond_gray = gray
    cond_rgb = cv2.cvtColor(cond_gray, cv2.COLOR_GRAY2RGB)
    return cond_gray, cond_rgb


def fold_angle(a):
    """fold angles beyond bounds back"""
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


def get_head_pose(landmarks, img_w, img_h):
    """Yaw, Pitch, Roll in degrees using solvePnP"""
    face_2d = []
    for idx in LANDMARK_INDICES_POSE:
        lm = landmarks[idx]
        face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
    face_2d = np.array(face_2d, dtype=np.float64)

    # Pi cam 3 noir wide has 102-degree horizontal FOV
    focal_length = img_w / (2.0 * np.tan(np.radians(102.0 / 2.0)))
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = fold_angle(angles[0] * -1)
    yaw = fold_angle(angles[1])
    roll = fold_angle(angles[2])
    return pitch, yaw, roll


def get_normalized_iris(landmarks, img_w, img_h):
    """Normalized horizontal/vertical iris position (0 to 1)"""
    p_outer = np.array([landmarks[33].x * img_w, landmarks[33].y * img_h])
    p_inner = np.array([landmarks[133].x * img_w, landmarks[133].y * img_h])
    p_top = np.array([landmarks[159].x * img_w, landmarks[159].y * img_h])
    p_bot = np.array([landmarks[145].x * img_w, landmarks[145].y * img_h])
    p_iris = np.array([landmarks[468].x * img_w, landmarks[468].y * img_h])

    eye_width = np.linalg.norm(p_inner - p_outer)
    eye_height = np.linalg.norm(p_bot - p_top)

    horiz_ratio = np.linalg.norm(p_iris - p_outer) / eye_width if eye_width > 0 else 0.5
    vert_ratio = np.linalg.norm(p_iris - p_top) / eye_height if eye_height > 0 else 0.5
    return horiz_ratio, vert_ratio


# =====================================================================
# MAIN LOOP
# =====================================================================
print(f"Starting DMS | Conditioning: {CONDITIONING_MODE} | "
      f"Threshold: {DISTRACTION_THRESHOLD_S}s")
print("-" * 85)

alerts.clear_alert()          # shows "Driver OK / Monitoring"

away_since = None             # time the driver first went "away" (None = attentive)
attentive_since = None        # time the driver first looks back (helps with debouncing)
alert_active = False
last_beep_time = 0.0

try:
    while True:
        loop_start = time.monotonic()
        raw_frame = capture_pi_frame()
        if raw_frame is None:
            continue
        img_h, img_w, _ = raw_frame.shape

        #IMAGE CONDITIONING
        gray_img, rgb_img = apply_conditioning(raw_frame, CONDITIONING_MODE)

        #HAAR CASCADE (JUST FOR COMPARISON)
        haar_str = "Haar: off"
        if RUN_HAAR:
            t0 = time.monotonic()
            haar_faces = haar_cascade.detectMultiScale(
                gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            haar_ms = (time.monotonic() - t0) * 1000
            haar_str = f"Haar: {'YES' if len(haar_faces) > 0 else 'NO '} ({haar_ms:4.1f}ms)"

        #MEDIAPIPE FACE MESH
        t1 = time.monotonic()
        mp_results = face_mesh.process(rgb_img)
        mp_ms = (time.monotonic() - t1) * 1000

        face_lost = True
        looking_away = False
        if mp_results.multi_face_landmarks:
            face_lost = False
            landmarks = mp_results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = get_head_pose(landmarks, img_w, img_h)
            iris_h, iris_v = get_normalized_iris(landmarks, img_w, img_h)
            looking_away = abs(yaw) > YAW_LIMIT_DEG or pitch <= PITCH_MIN_DEG or pitch >= PITCH_MAX_DEG

            print(f"[MP: YES ({mp_ms:4.1f}ms)] -> "
                  f"Pose(Y:{yaw:+5.1f} P:{pitch:+5.1f}) | "
                  f"Iris(H:{iris_h:.2f} V:{iris_v:.2f})")
        else:
            print(f"[MP: NO  ({mp_ms:4.1f}ms)] -> FACE LOST / OCCLUSION")

        #DISTRACTION TIMER + ALERTS
        now = time.monotonic()
        away = face_lost or looking_away

        if away:
            attentive_since = None # reset "looking back" timer
            if away_since is None:
                away_since = now
            
            away_for = now - away_since
            if away_for >= DISTRACTION_THRESHOLD_S:
                if (not alert_active) or (now - last_beep_time >= REPEAT_BEEP_S):
                    if face_lost:
                        alerts.alert("NO_FACE", "Face not found")
                    else:
                        alerts.alert("DISTRACTED", "Eyes on road")
                    alert_active = True
                    last_beep_time = now
        else:
            if attentive_since is None:
                attentive_since = now
            
            #only clear timer if driver has been looking back for at least .5s
            #helps with outlier frames that read as looking forward while driver is actually looking away
            if now - attentive_since >= 0.5:
                away_since = None
                if alert_active:
                    alerts.clear_alert()
                    alert_active = False

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0, 0.1 - elapsed))

except KeyboardInterrupt:
    print("\nStopping DMS...")
finally:
    cam_proc.terminate()
    alerts.close()