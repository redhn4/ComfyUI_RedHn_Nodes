import copy
import json
import math
import os
import random
import re
import subprocess
import urllib.request
import uuid
import wave

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import imageio_ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

VIDEO_CATEGORY = "RedHn/视频"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

def calculate_ltx_frames(duration_sec, fps):
    target_frames = round(duration_sec * fps)
    n = round((target_frames - 1) / 8)
    return max(9, int(n * 8 + 1))

def load_image_from_path(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image)[None,]

def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

def parse_prompt_lines(text):
    lines = []
    try:
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:].rsplit("```", 1)[0]
        parsed = json.loads(clean_text.strip())
        if isinstance(parsed, dict):
            lines = [str(v).strip() for v in parsed.values() if str(v).strip()]
        elif isinstance(parsed, list):
            lines = [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    if not lines:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines or [""]

def resolve_distributed_image(default_image, image_folder, current_index):
    output_image = default_image
    if image_folder and os.path.isdir(image_folder):
        valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        image_files = sorted(
            (os.path.join(image_folder, name) for name in os.listdir(image_folder) if name.lower().endswith(valid_exts)),
            key=lambda path: natural_sort_key(os.path.basename(path))
        )
        if image_files:
            target_idx = min(current_index, len(image_files) - 1)
            try:
                output_image = load_image_from_path(image_files[target_idx])
            except Exception as exc:
                print(f"[RedHn-Image Distribution Error] Failed to load image: {exc}")
    return output_image

def submit_prompt(prompt_data):
    payload = {"prompt": prompt_data, "client_id": str(uuid.uuid4())}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    urllib.request.urlopen(req)

class RedHn_MV_AudioSniffer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "连接完整原始音频。"}),
                "帧率": ("FLOAT", {"default": 24.0, "step": 1.0, "tooltip": "最终视频帧率，常用 24 / 25 / 30。"}),
                "单段秒数": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1, "tooltip": "期望的单段时长，系统会自动修正到适合 LTX 的帧数。"}),
            }
        }
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("总段数", "总时长(秒)", "提示词参考")
    FUNCTION = "sniff"
    CATEGORY = VIDEO_CATEGORY

    def sniff(self, 音频, 帧率, 单段秒数):
        waveform = 音频["waveform"]
        sample_rate = 音频["sample_rate"]
        total_seconds = waveform.shape[-1] / sample_rate
        valid_frames = calculate_ltx_frames(单段秒数, 帧率)
        exact_duration = valid_frames / 帧率
        total_chunks = int(math.ceil(total_seconds / exact_duration))
        llm_prompt = (
            f"我有一首总长 {total_seconds:.2f} 秒的音频。它被精确等分为 {total_chunks} 段，"
            f"除了最后一段，前面每一段时长均为 {exact_duration:.4f} 秒。\n\n"
            f"请为这 {total_chunks} 段视频依次撰写连续的电影镜头脚本。\n"
            f"要求：\n"
            f"1. 必须严格输出 {total_chunks} 行，每行对应一段。\n"
            f"2. 每一行只输出纯英文 Prompt，专注画面描述，不要包含行号及说明文本。"
        )
        return (total_chunks, total_seconds, llm_prompt)

class RedHn_MV_AutoQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "完整原始音频。"}),
                "提示词文本": ("STRING", {"multiline": True, "default": "", "tooltip": "填入多行提示词，或传入 JSON / 列表格式内容。"}),
                "帧率": ("FLOAT", {"default": 24.0, "step": 1.0, "tooltip": "需要与最终视频保存节点使用的帧率一致。"}),
                "单段秒数": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1, "tooltip": "与 dt音频测量 保持一致。"}),
                "自动排队": ("BOOLEAN", {"default": True, "tooltip": "开启后自动把剩余分段提交到 ComfyUI 队列。"}),
                "当前序号": ("INT", {"default": 0, "min": 0, "tooltip": "系统内部编号，初始保持 0。"}),
                "会话ID": ("STRING", {"default": "", "tooltip": "系统追踪 ID，通常留空。"}),
            },
            "optional": {
                "图片输入": ("IMAGE", {"tooltip": "可选，所有分段共用同一张图。"}),
                "图片文件夹路径": ("STRING", {"default": "", "tooltip": "可选，按段号顺序取图；不足时重复最后一张。"}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }
    RETURN_TYPES = ("AUDIO", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("子音频", "当前提示词", "当前序号", "总段数", "生成帧数", "帧率", "保存前缀", "会话ID", "当前图片")
    FUNCTION = "process"
    CATEGORY = VIDEO_CATEGORY

    def process(self, 音频, 提示词文本, 帧率, 单段秒数, 自动排队, 当前序号, 会话ID, 图片输入=None, 图片文件夹路径="", prompt=None, unique_id=None):
        waveform = 音频["waveform"]
        sample_rate = 音频["sample_rate"]
        total_seconds = waveform.shape[-1] / sample_rate
        valid_frames = calculate_ltx_frames(单段秒数, 帧率)
        exact_duration = valid_frames / 帧率
        total_chunks = int(math.ceil(total_seconds / exact_duration))
        lines = parse_prompt_lines(提示词文本)
        current_prompt = lines[min(当前序号, len(lines) - 1)]
        session_id = 会话ID or ""
        if 当前序号 == 0 or not session_id:
            session_id = str(uuid.uuid4().hex)[:8]
        save_prefix = f"RedHn_MV_Record_{session_id}/chunk"
        if 自动排队 and 当前序号 == 0 and prompt and unique_id:
            print(f"[RedHn-MV Queue] 正在后台提交后续 {total_chunks - 1} 段任务...")
            for i in range(1, total_chunks):
                try:
                    new_prompt = copy.deepcopy(prompt)
                    node_data = new_prompt[unique_id]
                    node_data["inputs"]["当前序号"] = i
                    node_data["inputs"]["自动排队"] = False
                    node_data["inputs"]["会话ID"] = session_id
                    submit_prompt(new_prompt)
                except Exception as exc:
                    print(f"[RedHn-MV Queue Error] 后台任务提交失败：{exc}")
        chunk_start_sec = 当前序号 * exact_duration
        chunk_end_sec = (当前序号 + 1) * exact_duration
        if 当前序号 == total_chunks - 1 and chunk_end_sec > total_seconds:
            last_chunk_duration = total_seconds - chunk_start_sec
            chunk_frames = calculate_ltx_frames(last_chunk_duration, 帧率)
            applied_end_sec = chunk_start_sec + (chunk_frames / 帧率)
        else:
            chunk_frames = valid_frames
            applied_end_sec = chunk_end_sec
        start_idx = int(chunk_start_sec * sample_rate)
        end_idx = int(applied_end_sec * sample_rate)
        chunk_waveform = waveform[:, :, start_idx:end_idx]
        chunk_audio = {"waveform": chunk_waveform, "sample_rate": sample_rate}
        output_image = resolve_distributed_image(图片输入, 图片文件夹路径, 当前序号)
        return (chunk_audio, current_prompt, 当前序号, total_chunks, chunk_frames, 帧率, save_prefix, session_id, output_image)

class RedHn_MV_BatchQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "总段数": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "需要系统批量生成多少段。"}),
                "生成帧数": ("INT", {"default": 121, "min": 1, "max": 2000, "tooltip": "每段固定输出帧数，例如 121 / 97 / 81。"}),
                "帧率": ("FLOAT", {"default": 24.0, "step": 1.0, "tooltip": "透传给最终视频保存节点的帧率。"}),
                "提示词文本": ("STRING", {"multiline": True, "default": "", "tooltip": "多行提示词文本。"}),
                "提示词模式": (["顺序兜底", "完全随机抽取"], {"tooltip": "顺序兜底：按顺序取词，不够时重复最后一行。完全随机抽取：每段随机取一条。"}),
                "自动排队": ("BOOLEAN", {"default": True, "tooltip": "开启后自动把剩余任务提交到后台队列。"}),
                "当前序号": ("INT", {"default": 0, "min": 0, "tooltip": "系统内部编号，初始保持 0。"}),
                "会话ID": ("STRING", {"default": "", "tooltip": "系统追踪 ID，通常留空。"}),
            },
            "optional": {
                "默认图片": ("IMAGE", {"tooltip": "可选，所有段落都使用同一张图。"}),
                "图片文件夹": ("STRING", {"default": "", "tooltip": "可选，按段落编号从文件夹中依次取图；不足时重复最后一张。"}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("当前底图", "当前提示词", "当前序号", "总段数", "生成帧数", "帧率", "保存前缀", "会话ID")
    FUNCTION = "process"
    CATEGORY = VIDEO_CATEGORY

    def process(self, 总段数, 生成帧数, 帧率, 提示词文本, 提示词模式, 自动排队, 当前序号, 会话ID, 默认图片=None, 图片文件夹="", prompt=None, unique_id=None):
        total_chunks = max(1, 总段数)
        lines = parse_prompt_lines(提示词文本)
        current_prompt = random.choice(lines) if 提示词模式 == "完全随机抽取" else lines[min(当前序号, len(lines) - 1)]
        session_id = 会话ID or ""
        if 当前序号 == 0 or not session_id:
            session_id = str(uuid.uuid4().hex)[:8]
        save_prefix = f"RedHn_Batch_Array_{session_id}/section"
        if 自动排队 and 当前序号 == 0 and prompt and unique_id:
            print(f"[RedHn-Batch Queue] 正在后台提交后续 {total_chunks - 1} 段任务...")
            for i in range(1, total_chunks):
                try:
                    new_prompt = copy.deepcopy(prompt)
                    node_data = new_prompt[unique_id]
                    node_data["inputs"]["当前序号"] = i
                    node_data["inputs"]["自动排队"] = False
                    node_data["inputs"]["会话ID"] = session_id
                    submit_prompt(new_prompt)
                except Exception as exc:
                    print(f"[RedHn-Batch Queue Error] 后台任务提交失败：{exc}")
        output_image = resolve_distributed_image(默认图片, 图片文件夹, 当前序号)
        return (output_image, current_prompt, 当前序号, total_chunks, 生成帧数, 帧率, save_prefix, session_id)

class RedHn_MV_VideoConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "保存信号": (any_type, {"tooltip": "连接视频保存节点的输出，用于追踪文件落盘。"}),
                "当前序号": ("INT", {"default": 0, "tooltip": "连接队列节点输出的当前序号。"}),
                "总段数": ("INT", {"default": 1, "tooltip": "连接总段数。"}),
                "会话ID": ("STRING", {"default": "", "tooltip": "连接同一个会话 ID。"}),
                "自动合并": ("BOOLEAN", {"default": True, "tooltip": "关闭时只做记录，不执行最终合并。"}),
                "输出名称": ("STRING", {"default": "RedHn_Output_MV", "tooltip": "最终合成视频的文件名。"}),
            },
            "optional": {
                "原始音频": ("AUDIO", {"tooltip": "可选，提供完整原始音频时，合并后会重新覆盖总音轨。"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("最终成片路径",)
    FUNCTION = "process"
    CATEGORY = VIDEO_CATEGORY
    OUTPUT_NODE = True

    def _extract_video_path(self, data):
        if isinstance(data, str) and data.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            return os.path.join(folder_paths.get_output_directory(), data)
        if isinstance(data, dict):
            for value in data.values():
                result = self._extract_video_path(value)
                if result:
                    return result
        if isinstance(data, (list, tuple)):
            for value in data:
                result = self._extract_video_path(value)
                if result:
                    return result
        return None

    def process(self, 保存信号, 当前序号, 总段数, 会话ID, 自动合并, 输出名称, 原始音频=None):
        actual_video_path = self._extract_video_path(保存信号)
        if not actual_video_path or not os.path.isfile(actual_video_path):
            print(f"[RedHn-Concat] 未找到已生成的视频文件：{保存信号}")
            return ("",)
        temp_dir = folder_paths.get_temp_directory()
        session_file = os.path.join(temp_dir, f"redhn_mv_sess_{会话ID}.json")
        state = {}
        if os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as file:
                    state = json.load(file)
            except Exception:
                pass
        state[str(当前序号)] = actual_video_path
        with open(session_file, "w", encoding="utf-8") as file:
            json.dump(state, file)
        print(f"[RedHn-Concat] 已记录第 {当前序号 + 1}/{总段数} 段：{os.path.basename(actual_video_path)}")
        if 当前序号 < 总段数 - 1:
            return ("",)
        if not 自动合并:
            print("[RedHn-Concat] 已关闭自动合并，所有片段已保留在输出目录。")
            if os.path.exists(session_file):
                os.unlink(session_file)
            return ("",)
        if not HAS_FFMPEG:
            raise RuntimeError("未检测到 FFmpeg 环境，请安装 imageio[ffmpeg]")
        ordered_files = []
        for i in range(总段数):
            path = state.get(str(i))
            if not path or not os.path.isfile(path):
                print(f"[RedHn-Concat Error] 缺少序号 {i} 对应的视频片段，无法合并。")
                return ("",)
            ordered_files.append(path)
        list_file_path = os.path.join(temp_dir, f"redhn_mv_concat_{会话ID}.txt")
        with open(list_file_path, "w", encoding="utf-8") as file:
            for path in ordered_files:
                safe_path = path.replace("\\", "/").replace("'", "'\\''")
                file.write(f"file '{safe_path}'\n")
        output_subfolder = os.path.join(folder_paths.get_output_directory(), f"RedHn_FinalMerge_{会话ID}")
        os.makedirs(output_subfolder, exist_ok=True)
        out_path = os.path.join(output_subfolder, f"{输出名称}_{会话ID}.mp4")
        temp_wav_path = None
        if 原始音频 is not None:
            waveform = 原始音频["waveform"].squeeze(0)
            sample_rate = 原始音频["sample_rate"]
            temp_wav_path = os.path.join(temp_dir, f"redhn_mv_audio_{会话ID}.wav")
            if waveform.dtype in [torch.float16, torch.float32, torch.float64]:
                waveform = torch.clamp(waveform, -1.0, 1.0)
                waveform = (waveform * 32767.0).to(torch.int16)
            interleaved = waveform.t().cpu().numpy()
            with wave.open(temp_wav_path, "wb") as wav_file:
                wav_file.setnchannels(waveform.shape[0])
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(interleaved.tobytes())
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if temp_wav_path and os.path.exists(temp_wav_path):
                cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-i", temp_wav_path,
                       "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path]
            else:
                cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", out_path]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if result.returncode != 0:
                print(f"[RedHn-Concat FFmpeg Error]\n{result.stderr}")
            else:
                print(f"[RedHn-Concat] 合并完成：{out_path}")
                if os.path.exists(session_file):
                    os.unlink(session_file)
                if os.path.exists(list_file_path):
                    os.unlink(list_file_path)
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
        except Exception as exc:
            print(f"[RedHn-Concat Error] {exc}")
        return (out_path,)

NODE_CLASS_MAPPINGS = {
    "RedHn音频测量": RedHn_MV_AudioSniffer,
    "RedHn音频自动切片": RedHn_MV_AutoQueue,
    "RedHn批量图像分发": RedHn_MV_BatchQueue,
    "RedHn视频合并": RedHn_MV_VideoConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RedHn音频测量": "RedHn音频测量",
    "RedHn音频自动切片": "RedHn音频自动切片",
    "RedHn批量图像分发": "RedHn批量图像分发",
    "RedHn视频合并": "RedHn视频合并",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']