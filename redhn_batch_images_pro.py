import os
import torch
from PIL import Image, ImageOps
import numpy as np

class RedHn_BatchImagesPro_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": "", "multiline": True}),
                "最大批次": ("INT", {"default": 1000, "min": 1, "max": 1000, "step": 1}),
                "数量模式": ("BOOLEAN", {"default": False, "label_on": "每文件夹", "label_off": "全局共计"}),
                "检索排序": (["由旧到新", "由新到旧", "文件名升序", "文件名降序"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("图像批次", "图像列表")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "load_images"
    CATEGORY = "RedHn"

    def load_images(self, 文件夹路径, 最大批次, 数量模式, 检索排序):
        folders = [line.strip() for line in 文件夹路径.splitlines() if line.strip()]
        if not folders:
            return (torch.zeros((0, 64, 64, 3)), [])

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        try:
            import pillow_jxl
            exts.add('.jxl')
        except ImportError:
            pass

        all_files = []
        for folder in folders:
            if not os.path.isdir(folder):
                print(f"[RedHn批量图像Pro] 跳过无效路径: {folder}")
                continue
            folder_files = []
            for root, _, files in os.walk(folder):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in exts):
                        folder_files.append(os.path.join(root, f))
            if not folder_files:
                continue

            # 当前文件夹内排序
            if 检索排序 == "由旧到新":
                folder_files.sort(key=lambda x: os.path.getmtime(x))
            elif 检索排序 == "由新到旧":
                folder_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            elif 检索排序 == "文件名升序":
                folder_files.sort(key=lambda x: os.path.basename(x).lower())
            elif 检索排序 == "文件名降序":
                folder_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)

            # 如果是每文件夹模式，截取前最大批次个
            if 数量模式:  # True = 每文件夹
                folder_files = folder_files[:最大批次]

            all_files.extend(folder_files)

        # 如果是全局共计模式，全局截取前最大批次个
        if not 数量模式 and len(all_files) > 最大批次:
            all_files = all_files[:最大批次]

        if not all_files:
            return (torch.zeros((0, 64, 64, 3)), [])

        images = []
        for file_path in all_files:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                rgb_img = img.convert("RGB")
                img_tensor = torch.from_numpy(np.array(rgb_img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
            except Exception as e:
                print(f"加载失败 {file_path}: {e}")

        if not images:
            return (torch.zeros((0, 64, 64, 3)), [])

        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        resized = []
        for img in images:
            img = img.permute(0, 3, 1, 2)
            img = torch.nn.functional.interpolate(img, size=(max_h, max_w), mode='bilinear', align_corners=False)
            img = img.permute(0, 2, 3, 1)
            resized.append(img)
        batch = torch.cat(resized, dim=0)

        return (batch, images)


class RedHn_BatchImagesPro_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": True}),
                "max_batch": ("INT", {"default": 1000, "min": 1, "max": 1000, "step": 1}),
                "mode": ("BOOLEAN", {"default": False, "label_on": "Per folder", "label_off": "Global total"}),
                "sort_order": (["Oldest first", "Newest first", "Name A-Z", "Name Z-A"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_batch", "image_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "load_images"
    CATEGORY = "RedHn"

    def load_images(self, folder_path, max_batch, mode, sort_order):
        folders = [line.strip() for line in folder_path.splitlines() if line.strip()]
        if not folders:
            return (torch.zeros((0, 64, 64, 3)), [])

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        try:
            import pillow_jxl
            exts.add('.jxl')
        except ImportError:
            pass

        all_files = []
        for folder in folders:
            if not os.path.isdir(folder):
                print(f"[RedHn Batch Images Pro] Skipping invalid path: {folder}")
                continue
            folder_files = []
            for root, _, files in os.walk(folder):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in exts):
                        folder_files.append(os.path.join(root, f))
            if not folder_files:
                continue

            if sort_order == "Oldest first":
                folder_files.sort(key=lambda x: os.path.getmtime(x))
            elif sort_order == "Newest first":
                folder_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            elif sort_order == "Name A-Z":
                folder_files.sort(key=lambda x: os.path.basename(x).lower())
            elif sort_order == "Name Z-A":
                folder_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)

            if mode:
                folder_files = folder_files[:max_batch]

            all_files.extend(folder_files)

        if not mode and len(all_files) > max_batch:
            all_files = all_files[:max_batch]

        if not all_files:
            return (torch.zeros((0, 64, 64, 3)), [])

        images = []
        for file_path in all_files:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                rgb_img = img.convert("RGB")
                img_tensor = torch.from_numpy(np.array(rgb_img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        if not images:
            return (torch.zeros((0, 64, 64, 3)), [])

        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        resized = []
        for img in images:
            img = img.permute(0, 3, 1, 2)
            img = torch.nn.functional.interpolate(img, size=(max_h, max_w), mode='bilinear', align_corners=False)
            img = img.permute(0, 2, 3, 1)
            resized.append(img)
        batch = torch.cat(resized, dim=0)

        return (batch, images)