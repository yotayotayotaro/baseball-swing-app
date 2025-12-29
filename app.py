import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import os
import csv
from collections import deque

# --- Streamlitページ設定 ---
st.set_page_config(page_title="野球スイング解析", layout="wide")

st.title("⚾ 野球スイング解析アプリ (Web版)")
st.write("自分のスイング動画をアップロードすると、AIが解析します。")

# --- 定数・クラス定義 (ロジックはv12と同じ) ---
BAT_LENGTH_CM = 85.0
MAX_GRIP_SPEED_KMH = 160.0 
MAX_HEAD_SPEED_KMH = 250.0
SMOOTHING_WINDOW = 5
DISPLAY_HEIGHT = 600

class MovingAverage:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
    def update(self, value):
        self.window.append(value)
        if not self.window: return 0.0
        return sum(self.window) / len(self.window)

class SwingPhysics:
    def __init__(self, height_cm, fps):
        self.height_cm = height_cm
        self.fps = fps
        self.estimated_shoulder_width = height_cm * 0.26
        self.prev_grip = None
        self.prev_angle = None
        self.smooth_grip = MovingAverage(SMOOTHING_WINDOW)
        self.smooth_head = MovingAverage(SMOOTHING_WINDOW)
        self.smooth_omega = MovingAverage(SMOOTHING_WINDOW)
        self.last_speeds = {'grip_kmh': 0, 'head_kmh': 0}

    def calculate(self, landmarks, width, height):
        if not landmarks: return self.last_speeds, None
        lm = landmarks.landmark
        def get_pt(idx): return np.array([lm[idx].x * width, lm[idx].y * height])

        r_sh, l_sh = get_pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER), get_pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
        r_wrist, l_wrist = get_pt(mp.solutions.pose.PoseLandmark.RIGHT_WRIST), get_pt(mp.solutions.pose.PoseLandmark.LEFT_WRIST)
        r_hip, l_hip = get_pt(mp.solutions.pose.PoseLandmark.RIGHT_HIP), get_pt(mp.solutions.pose.PoseLandmark.LEFT_HIP)

        shoulder_px = np.linalg.norm(r_sh - l_sh)
        if shoulder_px == 0: return self.last_speeds, None
        cm_per_pixel = self.estimated_shoulder_width / shoulder_px

        grip_pt = (r_wrist + l_wrist) / 2.0
        sh_center = (r_sh + l_sh) / 2.0
        hip_center = (r_hip + l_hip) / 2.0
        torso_center = 0.7 * sh_center + 0.3 * hip_center

        grip_speed_kmh = 0.0
        head_speed_kmh = 0.0
        vec_curr = grip_pt - torso_center
        angle_rad = math.atan2(vec_curr[1], vec_curr[0])

        if self.prev_grip is not None:
            dist_px = np.linalg.norm(grip_pt - self.prev_grip)
            if dist_px < shoulder_px * 0.8:
                dist_m = (dist_px * cm_per_pixel) / 100.0
                grip_speed_kmh = dist_m * self.fps * 3.6
            else:
                grip_speed_kmh = self.last_speeds['grip_kmh']

            if self.prev_angle is not None:
                angle_diff = abs(angle_rad - self.prev_angle)
                if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
                if angle_diff < 1.0:
                    omega_rad_s = angle_diff * self.fps
                    arm_radius_cm = np.linalg.norm(vec_curr) * cm_per_pixel
                    total_radius_m = (arm_radius_cm + BAT_LENGTH_CM) / 100.0
                    head_speed_kmh = grip_speed_kmh + (omega_rad_s * total_radius_m) * 3.6
                else:
                    head_speed_kmh = self.last_speeds['head_kmh']
            self.prev_angle = angle_rad

        if grip_speed_kmh > MAX_GRIP_SPEED_KMH: grip_speed_kmh = self.last_speeds['grip_kmh']
        if head_speed_kmh > MAX_HEAD_SPEED_KMH: head_speed_kmh = self.last_speeds['head_kmh']

        final_grip = self.smooth_grip.update(grip_speed_kmh)
        final_head = self.smooth_head.update(head_speed_kmh)
        self.prev_grip = grip_pt
        
        self.last_speeds = {'grip_kmh': int(final_grip), 'head_kmh': int(final_head)}
        return self.last_speeds, {'grip': (int(grip_pt[0]), int(grip_pt[1])), 'torso': (int(torso_center[0]), int(torso_center[1]))}

# --- サイドバー設定 ---
st.sidebar.header("設定")
height_input = st.sidebar.number_input("身長 (cm)", value=170.0, step=1.0)
uploaded_file = st.sidebar.file_uploader("動画をアップロード", type=["mp4", "mov"])

# --- メイン処理 ---
if uploaded_file is not None:
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # リサイズ設定
    scale_ratio = DISPLAY_HEIGHT / orig_h
    w = int(orig_w * scale_ratio)
    h = DISPLAY_HEIGHT
    
    # 解析用インスタンス
    physics = SwingPhysics(height_input, fps)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # 保存用設定
    output_filename = "processed_video.mp4"
    # H.264でエンコード (スマホ再生互換性のため)
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))
    
    st.write("解析中... しばらくお待ちください")
    progress_bar = st.progress(0)
    
    frame_count = 0
    traces = deque(maxlen=30)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # リサイズ
            frame = cv2.resize(frame, (w, h))
            
            # 解析
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            speeds, coords = physics.calculate(results.pose_landmarks, w, h)
            
            # 描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if coords:
                traces.append(coords['grip'])
                cv2.line(frame, coords['torso'], coords['grip'], (255,255,255), 1)
                cv2.circle(frame, coords['torso'], 5, (0,0,255), -1)
            
            if len(traces) > 1:
                for i in range(1, len(traces)):
                    cv2.line(frame, traces[i-1], traces[i], (0, 165, 255), 2)
            
            # 情報表示
            cv2.rectangle(frame, (0,0), (250, 100), (0,0,0), -1)
            cv2.putText(frame, f"Wrist: {speeds['grip_kmh']} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"Head : {speeds['head_kmh']} km/h", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            time_sec = frame_count / fps
            cv2.putText(frame, f"Time : {time_sec:.2f} s", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    
    # Streamlitで動画を表示 (コンバートが必要な場合があるため、簡易的にOpenCV出力をffmpegで変換すると確実ですが、今回は簡易実装)
    # ※注意: OpenCVのVideoWriterはWebブラウザ互換のmp4を作るのが苦手な場合があります。
    # 実際にはここで `ffmpeg -i processed_video.mp4 -vcodec libx264 out.mp4` のような変換を噛ませると確実です。
    
    st.success("解析完了！")
    
    # 簡易的に変換（ffmpegがインストールされている環境を想定）
    # スマホで見れる形式に変換
    os.system(f"ffmpeg -y -i {output_filename} -vcodec libx264 output_web.mp4")
    
    if os.path.exists("output_web.mp4"):
        video_file = open("output_web.mp4", 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        # CSVダウンロードボタンなどもここに追加可能
    else:
        st.error("動画変換に失敗しました。ffmpegがインストールされているか確認してください。")

else:
    st.info("サイドバーから動画をアップロードしてください。")