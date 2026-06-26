import torch
from PIL import Image, ImageDraw
import numpy as np

class RedHn_ExpandBorder_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "模式": (["像素值 (1-512)", "短边比 (1-50%)"], {"default": "短边比 (1-50%)"}),
                "扩展值": ("INT", {"default": 2, "min": 1, "max": 512, "step": 1}),
                "启用圆角": ("BOOLEAN", {"default": True, "label_on": "开", "label_off": "关"}),
                "圆角值": ("INT", {"default": 5, "min": 1, "max": 512, "step": 1}),
                "边框颜色": (["黑色", "白色"], {"default": "黑色"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "expand"
    CATEGORY = "RedHn"

    def expand(self, 图像, 模式, 扩展值, 启用圆角, 圆角值, 边框颜色):
        if 边框颜色 == "黑色":
            color_rgb = (0, 0, 0)
        else:
            color_rgb = (255, 255, 255)

        out_list = []
        for i in range(图像.shape[0]):
            single_img = 图像[i]
            pil_img = Image.fromarray((single_img.cpu().numpy() * 255).astype(np.uint8), mode='RGB')

            short_side = min(pil_img.width, pil_img.height)

            # 计算实际扩展像素
            if 模式 == "短边比 (1-50%)":
                ratio = min(扩展值, 50) / 100.0
                expand_px = int(round(short_side * ratio))
                expand_px = max(1, min(512, expand_px))
            else:  # 像素值 (1-512)
                expand_px = 扩展值

            # 计算实际圆角半径
            if 启用圆角:
                if 模式 == "像素值 (1-512)":
                    final_radius = 圆角值
                else:  # 短边比 (1-50%)
                    ratio = min(圆角值, 50) / 100.0
                    final_radius = int(round(short_side * ratio))
                    final_radius = max(1, min(512, final_radius))

                mask = Image.new('L', pil_img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle([(0, 0), pil_img.size], radius=final_radius, fill=255)
                bg = Image.new('RGB', pil_img.size, color_rgb)
                pil_img = Image.composite(pil_img, bg, mask)

            # 扩展边框
            new_w = pil_img.width + 2 * expand_px
            new_h = pil_img.height + 2 * expand_px
            canvas = Image.new('RGB', (new_w, new_h), color_rgb)
            canvas.paste(pil_img, (expand_px, expand_px))
            pil_img = canvas

            result_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            out_list.append(result_tensor.unsqueeze(0))

        out_batch = torch.cat(out_list, dim=0)
        return (out_batch,)


class RedHn_ExpandBorder_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Pixel value (1-512)", "Short side ratio (1-50%)"], {"default": "Short side ratio (1-50%)"}),
                "expand_value": ("INT", {"default": 2, "min": 1, "max": 512, "step": 1}),
                "enable_round": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "round_value": ("INT", {"default": 5, "min": 1, "max": 512, "step": 1}),
                "border_color": (["Black", "White"], {"default": "Black"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "expand"
    CATEGORY = "RedHn"

    def expand(self, image, mode, expand_value, enable_round, round_value, border_color):
        if border_color == "Black":
            color_rgb = (0, 0, 0)
        else:
            color_rgb = (255, 255, 255)

        out_list = []
        for i in range(image.shape[0]):
            single_img = image[i]
            pil_img = Image.fromarray((single_img.cpu().numpy() * 255).astype(np.uint8), mode='RGB')

            short_side = min(pil_img.width, pil_img.height)

            if mode == "Short side ratio (1-50%)":
                ratio = min(expand_value, 50) / 100.0
                expand_px = int(round(short_side * ratio))
                expand_px = max(1, min(512, expand_px))
            else:
                expand_px = expand_value

            if enable_round:
                if mode == "Pixel value (1-512)":
                    final_radius = round_value
                else:
                    ratio = min(round_value, 50) / 100.0
                    final_radius = int(round(short_side * ratio))
                    final_radius = max(1, min(512, final_radius))

                mask = Image.new('L', pil_img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle([(0, 0), pil_img.size], radius=final_radius, fill=255)
                bg = Image.new('RGB', pil_img.size, color_rgb)
                pil_img = Image.composite(pil_img, bg, mask)

            new_w = pil_img.width + 2 * expand_px
            new_h = pil_img.height + 2 * expand_px
            canvas = Image.new('RGB', (new_w, new_h), color_rgb)
            canvas.paste(pil_img, (expand_px, expand_px))
            pil_img = canvas

            result_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            out_list.append(result_tensor.unsqueeze(0))

        out_batch = torch.cat(out_list, dim=0)
        return (out_batch,)