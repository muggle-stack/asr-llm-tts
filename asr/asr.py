import wave
from pathlib import Path
import os
import time
import subprocess
import numpy as np

from .models.sensevoice_bin import SenseVoiceSmall
from .models.postprocess_utils import rich_transcription_postprocess

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
asr_model_dir = os.path.join(cache_dir, "sensevoice")
asr_model_path = os.path.join(asr_model_dir, "model_quant_optimized.onnx")
tar_path = os.path.join(cache_dir, "sensevoice.tar.gz")

Path(cache_dir).mkdir(parents=True, exist_ok=True)

Path(asr_model_dir).mkdir(parents=True, exist_ok=True)

class ASRModel:
    def __init__(self):
        if not os.path.exists(asr_model_path):
            print("模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")

        self._model_path = asr_model_dir
        self._model = SenseVoiceSmall(self._model_path, batch_size=10, quantize=True)

    def generate(self, audio_file):
        if isinstance(audio_file, np.ndarray):
            audio_path = audio_file
        if isinstance(audio_file, str):
            audio_path = [audio_file]
        asr_res = self._model(audio_path, language='zh', use_itn=True)
        asr_res = asr_res[0][0].tolist()
        text = rich_transcription_postprocess(asr_res[0])
        return text
