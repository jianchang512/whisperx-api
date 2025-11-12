# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "ffmpeg-python>=0.2.0",
#    "flask>=3.1.2",
#    "openai>=2.7.2",
#    "pydub>=0.25.1",
#    "waitress>=3.0.2",
#    "whisperx>=3.7.4",
# ]
# 
# [[tool.uv.index]]
# url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#    { index = "pytorch-cu128", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
#    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },  # darwin 为 macOS
# ]
# torchaudio = [
#    { index = "pytorch-cu128", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
#    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
# ]
#
# ///



import os
import tempfile
import torch
import whisperx
from flask import Flask, request, jsonify, render_template
from waitress import serve
import logging
import webbrowser
from threading import Timer
import shutil
import sys
import ffmpeg

# --- 全局配置与初始化 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_hf_token():
    """
    获取 Hugging Face 令牌。
    优先从当前目录的 'token.txt' 文件读取，如果失败则从环境变量 'HUGGING_FACE_TOKEN' 读取。
    """
    token = None
    token_file = 'token.txt'
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
            if token:
                logging.info(f"成功从 {token_file} 文件中读取 Hugging Face 令牌。")
                return token
        except Exception as e:
            logging.warning(f"无法从 {token_file} 读取令牌: {e}")

    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token:
        logging.info("成功从环境变量中读取 Hugging Face 令牌。")
    else:
        logging.warning("在 token.txt 或环境变量中均未找到 Hugging Face 令牌。说话人分离功能将被禁用。")
    return token

HF_TOKEN = get_hf_token()

# 设备和计算类型配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BATCH_SIZE = 16

logging.info(f"使用设备: {DEVICE}，计算类型: {COMPUTE_TYPE}")

# 模型配置
ALLOWED_MODELS = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo']
DEFAULT_MODEL = 'large-v3'

# 模型缓存
whisper_models_cache = {}
diarize_model = None
diarize_model_loaded = False

def get_whisper_model(model_name: str):
    if model_name not in whisper_models_cache:
        logging.info(f"正在加载 Whisper 模型 '{model_name}'...")
        try:
            model = whisperx.load_model(model_name, DEVICE, compute_type=COMPUTE_TYPE)
            whisper_models_cache[model_name] = model
            logging.info(f"模型 '{model_name}' 加载成功。")
        except Exception as e:
            logging.error(f"加载 Whisper 模型 '{model_name}' 失败: {e}")
            raise
    return whisper_models_cache[model_name]

def get_diarize_model():
    global diarize_model, diarize_model_loaded
    if not diarize_model_loaded:
        logging.info("正在尝试加载说话人分离模型...")
        if not HF_TOKEN:
            return None
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            diarize_model_loaded = True
            logging.info("说话人分离模型加载成功。")
        except Exception as e:
            logging.error(f"严重错误: 说话人分离模型加载失败。此功能将被禁用。错误信息: {e}")
            diarize_model = None 
            diarize_model_loaded = True
    return diarize_model

# --- Flask 应用 ---
app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/v1/audio/transcriptions', methods=['POST'])
def audio_transcriptions():
    if 'file' not in request.files:
        return jsonify({"error": "请求中未包含文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择任何文件"}), 400

    model_id = request.form.get('model', DEFAULT_MODEL)
    model_name = 'large-v3' if model_id == 'large-v3-turbo' else model_id
    if model_name not in ALLOWED_MODELS:
        model_name = DEFAULT_MODEL
    
    language = request.form.get('language') or None
    prompt = request.form.get('prompt')
    
    logging.info(f"收到请求: 模型='{model_id}', 语言='{language or '自动检测'}', 提示词='{'有' if prompt else '无'}'")

    input_file_path = None
    processed_wav_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            input_file_path = tmp.name

        logging.info(f"正在将上传的文件 '{file.filename}' 转换为标准的 16kHz 单声道 WAV 格式...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            processed_wav_path = tmp_wav.name
        
        try:
            (
                ffmpeg
                .input(input_file_path)
                .output(processed_wav_path, ac=1, ar=16000, acodec='pcm_s16le', vn=None)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            logging.info("文件格式转换成功。")
        except ffmpeg.Error as e:
            error_details = e.stderr.decode('utf-8', errors='ignore')
            logging.error(f"FFmpeg 文件转换失败: {error_details}")
            return jsonify({"error": f"音频/视频文件处理失败，可能是文件已损坏或格式不受支持。"}), 400

        audio = whisperx.load_audio(processed_wav_path)
        model = get_whisper_model(model_name)
        
        # ---
        # *** FIX IS HERE ***
        # ---
        transcribe_options = {}
        if language:
            transcribe_options['language'] = language
        if prompt:
            # 使用正确的参数名 'prompt'
            transcribe_options['prompt'] = prompt
            
        result = model.transcribe(audio, batch_size=BATCH_SIZE, **transcribe_options)
        
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        diar_model = get_diarize_model()
        if diar_model:
            try:
                diarize_segments = diar_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                logging.error(f"说话人分离运行时失败: {e}。将回退到单说话人模式。")
        
        speakers = {segment.get('speaker') for segment in result["segments"] if 'speaker' in segment}
        is_single_speaker = len(speakers) <= 1
        logging.info(f"检测到的说话人: {speakers}。单说话人模式: {'是' if is_single_speaker else '否'}")

        speaker_mapping = {f"SPEAKER_{i:02d}": f"Speaker{i+1}" for i in range(20)}

        formatted_segments = []
        for segment in result["segments"]:
            speaker_raw = segment.get("speaker", "SPEAKER_00")
            speaker_name = speaker_mapping.get(speaker_raw, speaker_raw)
            text = segment['text'].strip()
            if not text:
                continue

            segment_speaker = speaker_name if not is_single_speaker else None

            formatted_segments.append({
                "start": segment['start'],
                "end": segment['end'],
                "text": text,
                "speaker": segment_speaker
            })
        
        response_data = {"segments": formatted_segments}
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"处理流程中发生未知错误: {e}", exc_info=True)
        return jsonify({"error": "处理过程中发生内部错误。"}), 500
    finally:
        if input_file_path and os.path.exists(input_file_path):
            os.remove(input_file_path)
            logging.info(f"已清理临时上传文件: {input_file_path}")
        if processed_wav_path and os.path.exists(processed_wav_path):
            os.remove(processed_wav_path)
            logging.info(f"已清理临时WAV文件: {processed_wav_path}")

# --- 启动服务 ---
def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        logging.error("错误: 系统 PATH 中未找到 FFmpeg。")
        print("\n错误: 系统 PATH 中未找到 FFmpeg。")
        print("请确保您已安装 FFmpeg 并且其路径已添加到系统环境变量中。")
        print("Windows 安装指南: https://www.wikihow.com/Install-FFmpeg-on-Windows")
        print("macOS (使用 Homebrew): brew install ffmpeg")
        print("Linux (Ubuntu/Debian): sudo apt update && sudo apt install ffmpeg")
        sys.exit(1)
    logging.info("FFmpeg 环境检查通过。")

def open_browser(url):
    webbrowser.open_new(url)

if __name__ == '__main__':
    check_ffmpeg()
    host = '127.0.0.1'
    port = 9092
    url = f"http://{host}:{port}"
    get_diarize_model()
    Timer(1, lambda: open_browser(url)).start()
    logging.info(f"服务已启动，正在监听 http://{host}:{port}")
    serve(app, host=host, port=port, threads=10)