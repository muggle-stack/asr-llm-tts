# 本地 ASR、LLM、TTS

本地端到端语音 ASR-LLM-TTS Onnx pipeline，不使用一个模型实现这些功能。

## 说明

1. 优化

因为端到端本来就是要求低算力的情况下实现。目前市面上的大模型在端侧耗费的算力较高。拆分成三个小模型单独进行优化替换，找到最小算力下，最高精度的端到端实现。

2. 解藕

asr、llm、tts目前以 pipeline 的形式整合，每个项目都可以单独替换，或者基于功能快速开发融入其他项目。

3. 异步

线程 / 进程 / 队列 把三个 stage 串起来可以 流水线并行——ASR 还在转写第 N 句时，LLM 已在生成第 N-1 句的回复，TTS 在合成第 N-2 句，整体延迟压得更低。

4. 资源隔离

速度较慢的 tts 可以单独使用 GPU 推理，速度较快的 asr 则可以使用 CPU 进行推理。

5. 可扩展

音频采集可添加 VAD、情感检测、音量归一化、翻译等功能，只是多加一个节点，不用大改代码。

6. 异构

目前我的代码实现全部是 python，但是因为接口的兼容性，可以将三个模块使用其他代码实现，如产品阶段的 c++，可以分阶段搭建。

## 使用

克隆代码

```bash
git clone https://github.com/muggle-stack/asr-llm-tts.git
cd asr-llm-tts
```

下载 ollama (https://ollama.com/download/windows)
```bash
ollama pull qwen3:0.6b
```

推荐虚拟环境安装依赖，不想虚拟环境就跳过这点：
```bash
sudo apt install venv
python -m venv .venv
source ~/.venv/bin/activate
```

```bash
pip install -r requirements.txt
python asr_llm_tts.py
```
