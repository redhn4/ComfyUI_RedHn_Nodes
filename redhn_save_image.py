import os
import re
import json
import torch
import numpy as np
from PIL import Image, PngImagePlugin
import folder_paths
from datetime import datetime
import random

# 尝试导入 piexif 用于处理 JPG 的元数据嵌入
try:
    import piexif
except ImportError:
    piexif = None

class RedHn_SaveImage_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 第一行：模式切换，激活(True)为保存，未激活(False)为预览
                "模式切换": ("BOOLEAN", {"default": True, "label_on": "保存", "label_off": "预览"}),
                "图像": ("IMAGE",),
                "保存格式": (["JPG", "PNG", "WEBP", "BMP", "TGA", "TIF"], {"default": "JPG"}),
                "嵌入元数据": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
                "文件名前缀": ("STRING", {"default": "ComfyUI_%date:yyyyMMdd_hhmmss%", "multiline": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "RedHn"

    def parse_date_format(self, fmt):
        mapping = {'yyyy': '%Y', 'MM': '%m', 'dd': '%d', 'hh': '%H', 'HH': '%H', 'mm': '%M', 'ss': '%S'}
        for old, new in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            fmt = fmt.replace(old, new)
        return fmt

    def generate_base_filename(self, template, ext):
        def replace_date(match):
            inner = match.group(1)
            try:
                strftime_fmt = self.parse_date_format(inner)
                return datetime.now().strftime(strftime_fmt)
            except Exception:
                return datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = re.sub(r'%date:(.*?)%', replace_date, template)
        filename = "".join(c for c in filename if c.isalnum() or c in ' ._-')
        if not filename: filename = "RedHn_Image"
        return filename, ext.lower()

    def get_next_filename(self, base, ext, start_idx):
        output_dir = folder_paths.get_output_directory()
        idx = start_idx
        while True:
            fname = f"{base}_{idx:04d}.{ext}"
            if not os.path.exists(os.path.join(output_dir, fname)):
                return fname, idx
            idx += 1

    def save(self, 模式切换, 图像, 保存格式, 嵌入元数据, 文件名前缀, prompt=None, extra_pnginfo=None):
        results = []
        
        if 图像.dim() == 4:
            batch_size = 图像.shape[0]
        else:
            batch_size = 1
            图像 = 图像.unsqueeze(0)

        # --- 预览逻辑 ---
        if not 模式切换:
            temp_dir = folder_paths.get_temp_directory()
            timestamp = datetime.now().strftime('%H%M%S_%f') 
            for i in range(batch_size):
                img_tensor = 图像[i]
                img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                temp_name = f"redhn_preview_{timestamp}_{i}_{random.randint(1000,9999)}.png"
                pil_img.save(os.path.join(temp_dir, temp_name))
                results.append({"filename": temp_name, "subfolder": "", "type": "temp"})
            
            return {"ui": {"images": results}, "result": (图像,)}

        # --- 保存逻辑 ---
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        format_map = {"JPG": "JPEG", "PNG": "PNG", "WEBP": "WEBP", "BMP": "BMP", "TGA": "TGA", "TIF": "TIFF"}
        pil_format = format_map.get(保存格式, "JPEG")
        base_name, ext = self.generate_base_filename(文件名前缀, 保存格式)

        existing = []
        if os.path.exists(output_dir):
            pattern = re.compile(rf"{re.escape(base_name)}_(\d{{4}})\.{ext}")
            for f in os.listdir(output_dir):
                m = pattern.match(f)
                if m: existing.append(int(m.group(1)))
        next_idx = max(existing) + 1 if existing else 1

        # 准备元数据
        metadata = None
        if 嵌入元数据 and 保存格式 == "PNG":
            metadata = PngImagePlugin.PngInfo()
            if prompt is not None: metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items(): metadata.add_text(k, json.dumps(v))

        for i in range(batch_size):
            fname, used_idx = self.get_next_filename(base_name, ext, next_idx)
            filepath = os.path.join(output_dir, fname)
            next_idx = used_idx + 1

            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mode = "RGBA" if img_np.shape[-1] == 4 else "RGB"
            pil_img = Image.fromarray(img_np, mode=mode)

            save_kwargs = {}
            if 保存格式 == "PNG":
                if metadata: save_kwargs["pnginfo"] = metadata
                save_kwargs["compress_level"] = 4
            elif 保存格式 == "JPG":
                save_kwargs["quality"] = 95
                if 嵌入元数据 and piexif is not None and prompt is not None:
                    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: b"Unicode\x00\x00\x00" + json.dumps(prompt).encode("utf-8")}}
                    save_kwargs["exif"] = piexif.dump(exif_dict)
            
            try:
                pil_img.save(filepath, format=pil_format, **save_kwargs)
                results.append({"filename": fname, "subfolder": "", "type": "output"})
            except Exception as e:
                print(f"[RedHn Save] Error: {e}")

        return {"ui": {"images": results}, "result": (图像,)}

class RedHn_SaveImage_EN(RedHn_SaveImage_CN):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Mode_Switch": ("BOOLEAN", {"default": True, "label_on": "Save", "label_off": "Preview"}),
                "IMAGE": ("IMAGE",),
                "Format": (["JPG", "PNG", "WEBP", "BMP", "TGA", "TIF"], {"default": "JPG"}),
                "Embed_Metadata": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "Filename_Prefix": ("STRING", {"default": "ComfyUI_%date:yyyyMMdd_hhmmss%", "multiline": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)

    def save(self, Mode_Switch, IMAGE, Format, Embed_Metadata, Filename_Prefix, prompt=None, extra_pnginfo=None):
        return super().save(Mode_Switch, IMAGE, Format, Embed_Metadata, Filename_Prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)