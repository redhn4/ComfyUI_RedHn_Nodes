import torch

class RedHn_GridSplitter_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "行": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "列": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像列表",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "RedHn"

    def split(self, 图像, 行, 列):
        if 图像.dim() != 4:
            raise ValueError("输入必须是4维张量 [B, H, W, C]")
        B, H, W, C = 图像.shape
        if B != 1:
            print("警告：输入批次大于1，将只处理第一张图像")
            img = 图像[0:1]
        else:
            img = 图像

        # 计算每个单元格的尺寸
        cell_h = H // 行
        cell_w = W // 列
        if cell_h == 0 or cell_w == 0:
            raise ValueError(f"图像尺寸过小，无法拆分为 {行}x{列} 宫格")

        patches = []
        for i in range(行):
            for j in range(列):
                y_start = i * cell_h
                y_end = y_start + cell_h
                x_start = j * cell_w
                x_end = x_start + cell_w
                patch = img[:, y_start:y_end, x_start:x_end, :]
                patches.append(patch)
        return (patches,)


class RedHn_GridSplitter_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "cols": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "RedHn"

    def split(self, image, rows, cols):
        if image.dim() != 4:
            raise ValueError("Input must be 4D tensor [B, H, W, C]")
        B, H, W, C = image.shape
        if B != 1:
            print("Warning: batch size > 1, only first image will be processed")
            img = image[0:1]
        else:
            img = image

        cell_h = H // rows
        cell_w = W // cols
        if cell_h == 0 or cell_w == 0:
            raise ValueError(f"Image too small to split into {rows}x{cols} grid")

        patches = []
        for i in range(rows):
            for j in range(cols):
                y_start = i * cell_h
                y_end = y_start + cell_h
                x_start = j * cell_w
                x_end = x_start + cell_w
                patch = img[:, y_start:y_end, x_start:x_end, :]
                patches.append(patch)
        return (patches,)