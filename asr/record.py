import time
import threading
import numpy as np
from collections import deque
# from scipy.signal import resample
from pathlib import Path
import subprocess
import os

import pyaudio            # 录音
import onnxruntime as ort  # VAD

# -------- Silero VAD 封装（无 torch） -------- #
RATE_VAD     = 16000
WIN_SAMPLES  = 512             # 32 ms
CTX_SAMPLES  = 64
TRIG_ON, TRIG_OFF = 0.60, 0.35 # 触发 / 结束阈值

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
asr_model_dir = os.path.join(cache_dir, "sensevoice")
asr_model_path = os.path.join(asr_model_dir, "model_quant_optimized.onnx")
silero_model_path = os.path.join(asr_model_dir, "silero_vad.onnx")
tar_path = os.path.join(cache_dir, "sensevoice.tar.gz")

Path(cache_dir).mkdir(parents=True, exist_ok=True)

Path(asr_model_dir).mkdir(parents=True, exist_ok=True)

class SileroVAD:
    """NumPy 封装，输入 bytes(512*2) -> 概率 float"""
    def __init__(self):
        if not os.path.exists(silero_model_path):
            print("模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")

        self.sess  = ort.InferenceSession(silero_model_path, providers=['CPUExecutionProvider'])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx   = np.zeros((1, CTX_SAMPLES), dtype=np.float32)
        self.sr    = np.array(RATE_VAD, dtype=np.int64)

    def reset(self):
        self.state.fill(0)
        self.ctx.fill(0)

    def __call__(self, pcm_bytes: bytes) -> float:
        wav = (np.frombuffer(pcm_bytes, dtype=np.int16)
                 .astype(np.float32) / 32768.0)[np.newaxis, :]      # (1,512)

        x   = np.concatenate((self.ctx, wav), axis=1)                # (1,576)
        self.ctx = x[:, -CTX_SAMPLES:]

        prob, self.state = self.sess.run(
            None,
            {"input": x.astype(np.float32),
             "state": self.state,
             "sr":    self.sr}
        )
        return float(prob)

# ============ 录音管线，改用 SileroVAD ============ #
class RecAudioPipeLine:
    def __init__(self,
                 sld: int = 1,
                 max_time: int = 5,
                 device_index: int = None):
        """
        sld         : 静音多少秒停止
        max_time    : 最长录音时长
        device_index: pyaudio 输入设备编号（None=默认）
        """
        self._sld            = sld
        self.max_time_record = max_time
        self.frame_is_append = True

        # ---- 录音参数固定为 16k / 512samples 与 VAD 对齐 ---- #
        self.RATE       = RATE_VAD
        self.FRAME_SIZE = WIN_SAMPLES
        self.FORMAT     = pyaudio.paInt16
        self.CHANNELS   = 1

        self.pa     = pyaudio.PyAudio()
        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   input=True,
                                   frames_per_buffer=self.FRAME_SIZE,
                                   input_device_index=device_index)

        self.vad      = SileroVAD()        # <<< 改这里
        self.hist     = deque(maxlen=10)   # 300 ms 平滑阈值
        self.time_start = 0

    # ------------- 录音 + VAD ---------------- #
    def vad_audio(self):
        frames = []
        speech_detected = False
        last_speech_time = time.time()
        self.time_start  = time.time()

        try:
            while True:
                frame = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)

                # --- Silero 推理 ---
                p = self.vad(frame)
                self.hist.append(p)
                prob_avg = np.mean(self.hist)
                is_speech = prob_avg > TRIG_ON if not speech_detected else prob_avg > TRIG_OFF

                if is_speech:
                    last_speech_time = time.time()
                    if not speech_detected:
                        speech_detected = True
                        print("▶ 检测到语音，开始录制...")

                if self.frame_is_append:
                    frames.append(frame)

                # ----- 停止条件 -----
                now = time.time()
                if (speech_detected and
                    now - last_speech_time > self._sld):
                    print(f"⏹ 静音超过 {self._sld}s，停止录制")
                    break

                if speech_detected and (now - self.time_start) >= self.max_time_record:
                    print(f"⏹ 录音超过 {self.max_time_record}s，停止录制")
                    break

        finally:
            self.stream.stop_stream()

        # --------- 返回 numpy int16 波形 16 k --------- #
        if not frames:
            return None
        audio_np = np.frombuffer(b"".join(frames), dtype=np.int16)
        return audio_np

    def record_audio(self):
        self.stream.start_stream()
        return self.vad_audio()

# -------- 线程包装 -------- #
class RecAudioThreadPipeLine(RecAudioPipeLine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread = None
        self.audio_np = None      # numpy int16

    def start_recording(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._record_audio_thread, daemon=True)
            self.thread.start()

    def _record_audio_thread(self):
        self.audio_np = self.record_audio()

    def stop_recording(self):
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def get_audio(self):
        return self.audio_np

# ---------------- 测试 ---------------- #
if __name__ == "__main__":
    rec = RecAudioThreadPipeLine(sld=1, max_time=5)
    print("按 Enter 开始录音 ...")
    input()
    rec.start_recording()
    rec.thread.join()
    wav = rec.get_audio()
    print("录音完成，采样点数:", None if wav is None else wav.shape[0])

