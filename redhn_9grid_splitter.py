import torch

class RedHn_NineGridSplitter_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "前像素收缩": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "后像素收缩": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像列表",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "RedHn"

    def split(self, 图像, 前像素收缩, 后像素收缩):
        if 图像.dim() != 4:
            raise ValueError("输入必须是4维张量 [B, H, W, C]")
        B, H, W, C = 图像.shape
        if B != 1:
            print("警告：输入批次大于1，将只处理第一张图像")
            img = 图像[0:1]
        else:
            img = 图像

        # 前像素收缩（整体外框）
        outer_ratio = 前像素收缩 / 100.0
        outer_h = int(H * outer_ratio)
        outer_w = int(W * outer_ratio)
        if outer_h * 2 >= H or outer_w * 2 >= W:
            raise ValueError("前像素收缩过大，剩余区域为空")
        img_cropped = img[:, outer_h:H - outer_h, outer_w:W - outer_w, :]

        # 九等分
        new_h = img_cropped.shape[1]
        new_w = img_cropped.shape[2]
        cell_h = new_h // 3
        cell_w = new_w // 3
        if cell_h == 0 or cell_w == 0:
            raise ValueError("前像素收缩后图像过小，无法三等分")
        blocks = []
        for row in range(3):
            for col in range(3):
                y_start = row * cell_h
                y_end = y_start + cell_h
                x_start = col * cell_w
                x_end = x_start + cell_w
                block = img_cropped[:, y_start:y_end, x_start:x_end, :]
                blocks.append(block)

        # 后像素收缩（每个子图内框）
        inner_ratio = 后像素收缩 / 100.0
        results = []
        for block in blocks:
            bh, bw = block.shape[1], block.shape[2]
            shrink_h = int(bh * inner_ratio)
            shrink_w = int(bw * inner_ratio)
            if shrink_h * 2 >= bh or shrink_w * 2 >= bw:
                raise ValueError("后像素收缩过大，剩余区域为空")
            inner_block = block[:, shrink_h:bh - shrink_h, shrink_w:bw - shrink_w, :]
            results.append(inner_block)
        return (results,)


class RedHn_NineGridSplitter_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Front shrink": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "label": "Front shrink"}),
                "Rear shrink": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "label": "Rear shrink"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "RedHn"

    def split(self, image, **kwargs):
        # 兼容新旧参数名
        Front_shrink = kwargs.get("Front shrink", kwargs.get("Front_shrink", 0))
        Rear_shrink = kwargs.get("Rear shrink", kwargs.get("Rear_shrink", 0))

        if image.dim() != 4:
            raise ValueError("Input must be 4D tensor [B, H, W, C]")
        B, H, W, C = image.shape
        if B != 1:
            print("Warning: batch size > 1, only first image will be processed")
            img = image[0:1]
        else:
            img = image

        outer_ratio = Front_shrink / 100.0
        outer_h = int(H * outer_ratio)
        outer_w = int(W * outer_ratio)
        if outer_h * 2 >= H or outer_w * 2 >= W:
            raise ValueError("Front shrink too large, remaining area empty")
        img_cropped = img[:, outer_h:H - outer_h, outer_w:W - outer_w, :]

        new_h = img_cropped.shape[1]
        new_w = img_cropped.shape[2]
        cell_h = new_h // 3
        cell_w = new_w // 3
        if cell_h == 0 or cell_w == 0:
            raise ValueError("Image too small after front shrink")
        blocks = []
        for row in range(3):
            for col in range(3):
                y_start = row * cell_h
                y_end = y_start + cell_h
                x_start = col * cell_w
                x_end = x_start + cell_w
                block = img_cropped[:, y_start:y_end, x_start:x_end, :]
                blocks.append(block)

        inner_ratio = Rear_shrink / 100.0
        results = []
        for block in blocks:
            bh, bw = block.shape[1], block.shape[2]
            shrink_h = int(bh * inner_ratio)
            shrink_w = int(bw * inner_ratio)
            if shrink_h * 2 >= bh or shrink_w * 2 >= bw:
                raise ValueError("Rear shrink too large, remaining area empty")
            inner_block = block[:, shrink_h:bh - shrink_h, shrink_w:bw - shrink_w, :]
            results.append(inner_block)
        return (results,)