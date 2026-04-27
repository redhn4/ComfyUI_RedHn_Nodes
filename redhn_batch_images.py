import os
import torch
from PIL import Image, ImageOps
import numpy as np

class RedHn_BatchImages_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("图像批次", "图像列表")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "load_images"
    CATEGORY = "RedHn"

    def load_images(self, 文件夹路径):
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
                print(f"[RedHn批量图像] 跳过无效路径: {folder}")
                continue
            for root, _, files in os.walk(folder):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in exts):
                        all_files.append(os.path.join(root, f))

        if not all_files:
            return (torch.zeros((0, 64, 64, 3)), [])

        all_files.sort()
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

        # 统一尺寸生成批次
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


class RedHn_BatchImages_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_batch", "image_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "load_images"
    CATEGORY = "RedHn"

    def load_images(self, folder_path):
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
                print(f"[RedHn Batch Images] Skipping invalid path: {folder}")
                continue
            for root, _, files in os.walk(folder):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in exts):
                        all_files.append(os.path.join(root, f))

        if not all_files:
            return (torch.zeros((0, 64, 64, 3)), [])

        all_files.sort()
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