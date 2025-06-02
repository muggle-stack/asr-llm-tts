import os
import time
import subprocess

#from pydub import AudioSegment
import simpleaudio as sa
#from pydub.playback import play

from llm import LLMModel
llm_model = LLMModel()

from tts import TTSModel
tts_model = TTSModel()
warm_up = "欢迎使用端到端语音"
tts_model.ort_predict(warm_up)
print("All models init successfully!", flush=True)
from asr import ASRModel, RecAudioThreadPipeLine
asr_model = ASRModel()
rec_audio = RecAudioThreadPipeLine(sld=1, max_time=3, device_index=3)
import threading
import queue

output_queue = queue.Queue()
play_queue = queue.Queue()

def play_audio(file_path):
    try:
        if os.path.exists(file_path):
            #playsound(file_path)
            #subprocess.run(["afplay", file_path])
#            sound = AudioSegment.from_file(file_path)
#            play(sound)
            sa.WaveObject.from_wave_file(file_path).play().wait_done()
            #print("play_audio")
            #print(time.time())
        else:
            print(f"{file_path} does not exist")
    except Exception as e:
        print(f"An error occurred while trying to play the audio file: {e}")

def tts_thread():
    while True:
        try:
            index, text = output_queue.get()
            output_audio_path = tts_model.ort_predict(text)
            play_queue.put(output_audio_path)
            print("put_audio_to_queue", time.time())
        except Exception as e:
            print(f"An error occurred!{e}")

def play_thread():
    while True:
        try:
            output_audio = play_queue.get()
            play_audio(output_audio)
        except Exception as e:
            print(f"An error occurred!{e}")

def get_sentences_from_pos(start, text, sep='，。,.!?'):
    text_len = len(text)
    if text_len < start:
        raise Exception("start must be less then length of text")
    sentences = []
    last = start
    for i in range(start, text_len):
        if text[i] in sep:
            sentences.append(text[last:i+1])
            last = i + 1
            print("sentences", sentences[-1])
    return last, sentences

if __name__ == '__main__':
    tts_thread_instance = threading.Thread(target=tts_thread, daemon=True)
    tts_thread_instance.start()

    play_thread_instance = threading.Thread(target=play_thread, daemon=True)
    play_thread_instance.start()
    try:
        while True:
            #output_llm = ''
            print("Press enter to start!")
            input() # enter 触发
            rec_audio.start_recording()

            rec_audio.thread.join() # 等待录音完成
            audio_ret = rec_audio.get_audio() 
 
            text = asr_model.generate(audio_ret)
            print('user: ', text)

            t1 = time.time()
            llm_output = llm_model.generate(text)
            full_response = ''
            index = 0
            start = 0
            for output_text in llm_output:
                full_response += output_text
                last, sentences = get_sentences_from_pos(start, full_response)
                for s in sentences:
                    index += 1
                    output_queue.put((index, s))
                start = last
            if start < len(full_response):
                index += 1
                output_queue.put((index, full_response[start:]))
                #print(output_text, end='', flush=True)
            if full_response is None:
                continue
            
            #output_audio = tts_model.ort_predict(output_llm)
            #subprocess.run(["afplay", output_audio])

    except KeyboardInterrupt:
        print("process was interrupted by user.")

