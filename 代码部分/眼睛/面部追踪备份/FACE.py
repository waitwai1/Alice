import cv2
import serial
import time
import mediapipe as mp
import face_recognition
import os
import threading
import queue
import pickle

# ==================== 设置区 ====================
SERVO_PORT = "COM4"
CAMERA_INDEX = 1

# KP_PAN / KD_PAN：左右转动的灵敏度参数，数值越大，舵机越容易跟着目标快速修正偏差。
KP_PAN = 0.09
KD_PAN = 0.1
# KP_TILT / KD_TILT：上下转动的灵敏度参数，控制头部上下跟踪的响应速度。
KP_TILT = 0.06
KD_TILT = 0.035

# ALPHA：过滤噪声的平滑系数，数值较大时更容易跟随瞬时变化，较小时更稳定。
ALPHA = 0.35
# SPEED_MIN：舵机最低转速下限，数值越大，舵机越不容易“拖着走”，适合让目标快速响应和转向更明显。
SPEED_MIN = 60             # 设定最小速度下限，避免舵机在接近目标时转得太慢
# SPEED_MAX：舵机最高转速上限，过大可能导致动作过猛，过小则跟踪迟缓。
SPEED_MAX = 120            # 放宽上限
# PAN_DIR / TILT_DIR：控制左右/上下转动方向，决定正反转。
PAN_DIR  = -1
TILT_DIR = +1

# 非线性加速：误差超过这个值时，速度再乘以增益，帮助高速目标更快跟上。
ACCEL_THRESHOLD = 80       # 像素
ACCEL_GAIN = 1.1           # 加速倍数

KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_encodings.pkl"
TOLERANCE = 0.65
RECOGNIZE_INTERVAL = 0.8
MAX_FACES = 3

LOCK_TIMEOUT = 5.0         # 脸消失5秒内不解锁
MATCH_DISTANCE = 350       # 允许脸大范围移动
# ================================================

# ==================== 加载人脸档案 ====================
known_face_encodings = []
known_face_names = []

def build_cache():
    encodings = []
    names = []
    print("正在从照片提取人脸编码（首次运行较慢）...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encs = face_recognition.face_encodings(image)
            if len(encs) > 0:
                encodings.append(encs[0])
                names.append(os.path.splitext(filename)[0])
                print(f"  ✅ 已登记: {names[-1]}")
            else:
                print(f"  ⚠️ {filename} 中未检测到人脸，已跳过")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"✅ 缓存已保存到 {CACHE_FILE}")
    return encodings, names

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"⚠️ 未找到 {KNOWN_FACES_DIR} 文件夹")
    input("按回车退出...")
    exit()

need_rebuild = False
if not os.path.exists(CACHE_FILE):
    need_rebuild = True
else:
    cache_time = os.path.getmtime(CACHE_FILE)
    for f in os.listdir(KNOWN_FACES_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(KNOWN_FACES_DIR, f)
            if os.path.getmtime(fpath) > cache_time:
                need_rebuild = True
                break

if need_rebuild:
    known_face_encodings, known_face_names = build_cache()
else:
    print(f"✅ 从缓存加载人脸档案: {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
    for name in known_face_names:
        print(f"  ✅ 已加载: {name}")

print(f"共登记了 {len(known_face_names)} 个人脸。")
if len(known_face_names) == 0:
    input("没有可用的人脸档案，按回车退出...")
    exit()

# ==================== 连接Arduino ====================
try:
    ser = serial.Serial(SERVO_PORT, 115200, timeout=0.1)
    time.sleep(2)
    print(f"✅ 已连接Arduino: {SERVO_PORT}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    input("按回车退出...")
    exit()

# ==================== 打开摄像头 ====================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

if not cap.isOpened():
    print("❌ 摄像头打不开")
    input("按回车退出...")
    exit()

# ==================== MediaPipe ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

CX = 320
CY = 240

# ==================== PID控制器 ====================
class PIDController:
    def __init__(self, kp, kd, alpha=0.3):
        self.kp = kp
        self.kd = kd
        self.alpha = alpha
        self.prev_error = 0
        self.filtered_d = 0
    def compute(self, error, dt):
        p_out = self.kp * error
        raw_d = error - self.prev_error
        self.filtered_d = self.alpha * raw_d + (1 - self.alpha) * self.filtered_d
        d_out = self.kd * self.filtered_d
        self.prev_error = error
        return p_out + d_out
    def reset(self):
        self.prev_error = 0
        self.filtered_d = 0

pid_pan = PIDController(KP_PAN, KD_PAN, ALPHA)
pid_tilt = PIDController(KP_TILT, KD_TILT, ALPHA)

# ==================== 后台识别线程 ====================
recognize_queue = queue.Queue(maxsize=1)
recognize_result = {"nose_x": -1, "nose_y": -1, "time": 0}

def recognition_worker():
    while True:
        try:
            faces_data = recognize_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if faces_data is None:
            break
        try:
            for face in faces_data:
                rgb_face = cv2.cvtColor(face["roi"], cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)
                if len(encodings) > 0:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, encodings[0], tolerance=TOLERANCE
                    )
                    if True in matches:
                        recognize_result["nose_x"] = face["nose_x"]
                        recognize_result["nose_y"] = face["nose_y"]
                        recognize_result["time"] = time.time()
                        break
        except Exception:
            pass

worker = threading.Thread(target=recognition_worker, daemon=True)
worker.start()
print("✅ 后台识别线程已启动")

# ==================== 主循环 ====================
print("✅ 系统已就绪，开始追踪...")
print("按键盘 'q' 键退出")

last_time = time.time()
last_recognize_time = 0

# 位置锁定状态（关键改进：脸消失时也保持锁定）
is_locked = False
locked_nose_x = 0
locked_nose_y = 0
last_locked_time = 0
last_known_direction = 0   # 记录最后移动方向，用于搜索

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # ================== 有脸的情况 ==================
    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        faces_info = []

        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min = max(0, int(min(xs) * w) - 30)
            x_max = min(w, int(max(xs) * w) + 30)
            y_min = max(0, int(min(ys) * h) - 30)
            y_max = min(h, int(max(ys) * h) + 30)

            faces_info.append({
                "nose_x": nose_x,
                "nose_y": nose_y,
                "box": (x_min, y_min, x_max, y_max)
            })

        # ---- 送后台识别 ----
        if current_time - last_recognize_time > RECOGNIZE_INTERVAL:
            faces_to_send = []
            for f in faces_info:
                x_min, y_min, x_max, y_max = f["box"]
                if x_max > x_min and y_max > y_min:
                    roi = frame[y_min:y_max, x_min:x_max].copy()
                    roi = cv2.resize(roi, (150, 150))
                    faces_to_send.append({
                        "roi": roi,
                        "nose_x": f["nose_x"],
                        "nose_y": f["nose_y"]
                    })
            if faces_to_send:
                try:
                    recognize_queue.put_nowait(faces_to_send)
                except queue.Full:
                    pass
            last_recognize_time = current_time

        # ---- 首次锁定 ----
        if not is_locked and current_time - recognize_result["time"] < 2.0:
            min_dist = MATCH_DISTANCE
            best_idx = -1
            for i, f in enumerate(faces_info):
                dist = ((f["nose_x"] - recognize_result["nose_x"])**2 +
                        (f["nose_y"] - recognize_result["nose_y"])**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if best_idx >= 0:
                is_locked = True
                locked_nose_x = faces_info[best_idx]["nose_x"]
                locked_nose_y = faces_info[best_idx]["nose_y"]
                last_locked_time = current_time
                print("🔒 已锁定用户")

        # ---- 已锁定：找离锁定位置最近的脸 ----
        user_face_idx = -1
        if is_locked:
            min_dist = MATCH_DISTANCE
            for i, f in enumerate(faces_info):
                dist = ((f["nose_x"] - locked_nose_x)**2 +
                        (f["nose_y"] - locked_nose_y)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    user_face_idx = i
            if user_face_idx >= 0:
                # 更新锁定位置
                locked_nose_x = faces_info[user_face_idx]["nose_x"]
                locked_nose_y = faces_info[user_face_idx]["nose_y"]
                last_locked_time = current_time

        # ---- 画框 ----
        for i, f in enumerate(faces_info):
            x_min, y_min, x_max, y_max = f["box"]
            if i == user_face_idx:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, "ME", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, "Other", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # ---- 追踪 ----
        if user_face_idx >= 0:
            f = faces_info[user_face_idx]
            face_x = f["nose_x"]
            face_y = f["nose_y"]
            dx = face_x - CX
            dy = face_y - CY
            if abs(dx) < 15: dx = 0
            if abs(dy) < 15: dy = 0

            # 记录移动方向（用于脸消失时的搜索）
            if abs(dx) > 20:
                last_known_direction = dx

            adjust_pan = pid_pan.compute(dx, dt)
            adjust_tilt = pid_tilt.compute(dy, dt)

            # 【关键】非线性加速：误差越大，速度越快
            if abs(dx) > ACCEL_THRESHOLD:
                adjust_pan *= ACCEL_GAIN
            if abs(dy) > ACCEL_THRESHOLD:
                adjust_tilt *= ACCEL_GAIN

            pan_speed = 90 + PAN_DIR * adjust_pan
            tilt_speed = 90 + TILT_DIR * adjust_tilt
            pan_speed = max(SPEED_MIN, min(SPEED_MAX, pan_speed))
            tilt_speed = max(SPEED_MIN, min(SPEED_MAX, tilt_speed))

            ser.write(f"P{int(pan_speed)}T{int(tilt_speed)}\n".encode())

            cv2.circle(frame, (face_x, face_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (CX, CY), 5, (255, 0, 0), -1)
        else:
            # 有脸但不是你 → 停下
            ser.write(b"P90T90\n")
            pid_pan.reset()
            pid_tilt.reset()

    # ================== 没脸的情况 ==================
    else:
        # 【关键改进】已锁定时，脸消失不解除锁定，只让舵机停下
        # 保持锁定状态，等脸回来
        if is_locked and current_time - last_locked_time > LOCK_TIMEOUT:
            is_locked = False
            print("🔓 锁定超时，已解除")

        ser.write(b"P90T90\n")
        pid_pan.reset()
        pid_tilt.reset()

        # 显示锁定状态
        if is_locked:
            cv2.putText(frame, f"Searching... (locked)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Face Tracking (Only Me)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================== 清理 ====================
recognize_queue.put(None)
ser.write(b"P90T90\n")
cap.release()
cv2.destroyAllWindows()
ser.close()
print("已退出")