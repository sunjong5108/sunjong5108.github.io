# game_record.py
import cv2
import numpy as np
import mss
import time
import logging
import win32gui
import win32con
import ctypes
from collections import deque
import base64
import ray

logging.basicConfig(level=logging.INFO)

def enum_window_callback(hwnd, window_list):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title:
            window_list.append((hwnd, title))

def list_capture_windows():
    windows: list = []
    win32gui.EnumWindows(enum_window_callback, windows)
    return windows

def find_window_by_keyword(keyword):
    windows = list_capture_windows()
    for hwnd, title in windows:
        if keyword.lower() in title.lower():
            return hwnd, title
    return None, None

@ray.remote
class FrameBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)
        self.current_id = 0

    def put_frame(self, frame):
        ref = ray.put(frame)  # 객체 저장소에 프레임 저장
        self.buffer.append((self.current_id, ref))
        self.current_id += 1
        return self.current_id - 1

    def get_recent_refs(self, last_id=-1):
        if last_id == -1 and self.buffer:
            return self.buffer[-1][0], [self.buffer[-1][1]]
        return [(i, ref) for i, ref in self.buffer if i > last_id]

class GameRecorder:
    def __init__(self, window_title, output_file='arma3_recording.mp4', fps=20.0):
        self.window_title = window_title
        self.output_file = output_file
        self.fps = fps
        self.recording = False
        self.client_rect = None
        self.video_writer = None
        self.frame_buffer = FrameBuffer.options(name="global_buffer", namespace="make_sa_data", lifetime="detached").remote()  # 프레임 버퍼 인스턴스

    def get_client_rect(self):
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            hwnd, actual_title = find_window_by_keyword(self.window_title)
            if hwnd is None:
                possible_titles = [title for _, title in list_capture_windows()]
                raise Exception(f"Window not found: '{self.window_title}'. Available: {possible_titles}")
            else:
                logging.info("Using window '%s' (HWND: %s)", actual_title, hwnd)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            logging.warning("SetForegroundWindow error: %s", e)
        try:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except Exception as e:
            logging.warning("SetWindowPos error: %s", e)
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        client_rect = win32gui.GetClientRect(hwnd)
        right, bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
        return (left, top, right, bottom)

    def start_recording(self, duration=10, should_stop=None):
        self.client_rect = self.get_client_rect()
        sct = mss.mss()
        left, top, right, bottom = self.client_rect
        width = right - left
        height = bottom - top

        target_width, target_height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (target_width, target_height))

        ctypes.windll.user32.ShowCursor(0)
        self.recording = True
        start_time = time.time()
        while self.recording:
            # 중단 콜백이 주어졌으면 확인
            if should_stop is not None and should_stop():
                logging.info("should_stop() returned True, breaking recording loop.")
                break
            frame = sct.grab({"left": left, "top": top, "width": width, "height": height})
            img = np.array(frame)
            # ray.get(self.frame_buffer.put_frame.remote(img))  
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if (width, height) != (target_width, target_height):
                img = cv2.resize(img, (target_width, target_height))
            ret, buffer = cv2.imencode('.jpg', img)
            if ret:
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            ray.get(self.frame_buffer.put_frame.remote(jpg_as_text))   
            try:
                self.video_writer.write(img)
            except cv2.error as e:
                logging.error("OpenCV error during write: %s", e)
                break
            if time.time() - start_time > duration:
                break
        self.stop_recording()

    def stop_recording(self):
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
        ctypes.windll.user32.ShowCursor(1)
    
    # 새로 추가한 메서드: 다른 모듈(test_ray.py)에서 호출하여 최신 프레임을 가져올 수 있습니다.
    def get_latest_frame(self, timeout=1.0):
        return ray.put(self.frame_buffer.get(block=False, timeout=timeout))
