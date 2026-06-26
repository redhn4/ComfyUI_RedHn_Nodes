import torch

class RedHn_PixelShrink_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "像素收缩": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "shrink"
    CATEGORY = "RedHn"

    def shrink(self, 图像, 像素收缩):
        if 图像.dim() != 4:
            raise ValueError("输入必须是4维张量 [B, H, W, C]")
        B, H, W, C = 图像.shape
        if B != 1:
            print("警告：输入批次大于1，将只处理第一张图像")
            img = 图像[0:1]
        else:
            img = 图像

        ratio = 像素收缩 / 100.0
        shrink_h = int(H * ratio)
        shrink_w = int(W * ratio)
        if shrink_h * 2 >= H or shrink_w * 2 >= W:
            raise ValueError("像素收缩过大，剩余区域为空")
        img_cropped = img[:, shrink_h:H - shrink_h, shrink_w:W - shrink_w, :]
        return (img_cropped,)


class RedHn_PixelShrink_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_shrink": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "shrink"
    CATEGORY = "RedHn"

    def shrink(self, image, pixel_shrink):
        if image.dim() != 4:
            raise ValueError("Input must be 4D tensor [B, H, W, C]")
        B, H, W, C = image.shape
        if B != 1:
            print("Warning: batch size > 1, only first image will be processed")
            img = image[0:1]
        else:
            img = image

        ratio = pixel_shrink / 100.0
        shrink_h = int(H * ratio)
        shrink_w = int(W * ratio)
        if shrink_h * 2 >= H or shrink_w * 2 >= W:
            raise ValueError("Pixel shrink too large, remaining area empty")
        img_cropped = img[:, shrink_h:H - shrink_h, shrink_w:W - shrink_w, :]
        return (img_cropped,)