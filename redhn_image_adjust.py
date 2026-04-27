import torch
import torch.nn.functional as F

class RedHn_ImageAdjust_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "启用调整": ("BOOLEAN", {"default": True, "label_on": "开", "label_off": "关"}),
                "曝光": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "亮度": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "对比度": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "高光": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "阴影": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "自然饱和度": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色温": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色调": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "锐化": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "噪点": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "褪色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "adjust"
    CATEGORY = "RedHn"

    def adjust(self, 图像, 启用调整, 曝光, 亮度, 对比度, 高光, 阴影, 饱和度, 自然饱和度, 色温, 色调, 锐化, 噪点, 褪色):
        if not 启用调整:
            return (图像,)

        img = 图像.clone()
        B, H, W, C = img.shape
        img = img.permute(0, 3, 1, 2)  # [B, C, H, W]

        def param_to_factor(val, scale=2.0):
            return 1.0 + val / 100.0 * scale

        # 曝光
        if 曝光 != 0:
            exp_factor = 2.0 ** (曝光 / 100.0)
            img = img * exp_factor

        # 亮度
        if 亮度 != 0:
            img = img + 亮度 / 100.0

        # 对比度
        if 对比度 != 0:
            factor = param_to_factor(对比度, scale=1.0)
            img = (img - 0.5) * factor + 0.5

        # 高光
        if 高光 != 0:
            factor = param_to_factor(高光, scale=0.5)
            lum = img.mean(dim=1, keepdim=True)
            mask = (lum > 0.7).float()
            img = img + (img - 0.5) * (factor - 1.0) * mask * 0.5

        # 阴影
        if 阴影 != 0:
            factor = param_to_factor(阴影, scale=0.5)
            lum = img.mean(dim=1, keepdim=True)
            mask = (lum < 0.3).float()
            img = img + (0.5 - img) * (factor - 1.0) * mask * 0.5

        # 饱和度
        if 饱和度 != 0:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            max_rgb, _ = torch.max(img, dim=1, keepdim=True)
            min_rgb, _ = torch.min(img, dim=1, keepdim=True)
            s = (max_rgb - min_rgb) / (max_rgb + 1e-7)
            s = torch.clamp(s, 0, 1)
            factor = param_to_factor(饱和度, scale=1.0)
            img = (img - 0.5) * (1 + (factor - 1) * s) + 0.5

        # 自然饱和度
        if 自然饱和度 != 0:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            max_rgb, _ = torch.max(img, dim=1, keepdim=True)
            min_rgb, _ = torch.min(img, dim=1, keepdim=True)
            s = (max_rgb - min_rgb) / (max_rgb + 1e-7)
            factor = param_to_factor(自然饱和度, scale=1.0)
            weight = s
            img = (img - 0.5) * (1 + (factor - 1) * weight) + 0.5

        # 色温
        if 色温 != 0:
            factor = param_to_factor(色温, scale=0.5)
            img[:, 0:1] = img[:, 0:1] * (1 + max(0, factor - 1) * 0.5)
            img[:, 2:3] = img[:, 2:3] * (1 + max(0, 1 - factor) * 0.5)

        # 色调
        if 色调 != 0:
            factor = param_to_factor(色调, scale=0.3)
            img[:, 1:2] = img[:, 1:2] * factor

        # 锐化
        if 锐化 != 0:
            sharpen_factor = 锐化 / 100.0
            kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=img.dtype, device=img.device)
            kernel = kernel.repeat(3, 1, 1, 1)
            laplacian = F.conv2d(img, kernel, padding=1, groups=3)
            img = img + laplacian * sharpen_factor

        # 噪点调节：正值添加噪声（线性强度），负值降噪（强度随绝对值增大增强）
        if 噪点 != 0:
            if 噪点 > 0:
                noise_strength = 噪点 / 100.0
                noise = torch.randn_like(img) * noise_strength
                img = img + noise
            else:
                strength = abs(噪点) / 100.0  # 0.01 ~ 1.0
                # 根据强度决定滤波类型和次数
                if strength < 0.2:
                    kernel_size = 3
                    repeat = 1
                elif strength < 0.5:
                    kernel_size = 5
                    repeat = 1
                else:
                    kernel_size = 5
                    repeat = 2
                kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=img.dtype, device=img.device) / (kernel_size * kernel_size)
                kernel = kernel.repeat(C, 1, 1, 1)
                for _ in range(repeat):
                    img = F.conv2d(img, kernel, padding=kernel_size//2, groups=C)
            img = torch.clamp(img, 0, 1)

        # 褪色
        if 褪色 != 0:
            fade_factor = 褪色 / 100.0
            img = img * (1 - fade_factor * 0.5) + 0.5 * fade_factor * 0.5

        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 1)
        return (img,)


class RedHn_ImageAdjust_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_adjust": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "exposure": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "contrast": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "highlights": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadows": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "saturation": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "vibrance": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "temperature": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "tint": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "sharpen": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "noise": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "fade": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "RedHn"

    def adjust(self, image, enable_adjust, exposure, brightness, contrast, highlights, shadows, saturation, vibrance, temperature, tint, sharpen, noise, fade):
        if not enable_adjust:
            return (image,)

        img = image.clone()
        B, H, W, C = img.shape
        img = img.permute(0, 3, 1, 2)

        def param_to_factor(val, scale=2.0):
            return 1.0 + val / 100.0 * scale

        if exposure != 0:
            img = img * (2.0 ** (exposure / 100.0))
        if brightness != 0:
            img = img + brightness / 100.0
        if contrast != 0:
            factor = param_to_factor(contrast, scale=1.0)
            img = (img - 0.5) * factor + 0.5
        if highlights != 0:
            factor = param_to_factor(highlights, scale=0.5)
            lum = img.mean(dim=1, keepdim=True)
            mask = (lum > 0.7).float()
            img = img + (img - 0.5) * (factor - 1.0) * mask * 0.5
        if shadows != 0:
            factor = param_to_factor(shadows, scale=0.5)
            lum = img.mean(dim=1, keepdim=True)
            mask = (lum < 0.3).float()
            img = img + (0.5 - img) * (factor - 1.0) * mask * 0.5
        if saturation != 0:
            max_rgb = torch.max(img, dim=1, keepdim=True)[0]
            min_rgb = torch.min(img, dim=1, keepdim=True)[0]
            s = (max_rgb - min_rgb) / (max_rgb + 1e-7)
            factor = param_to_factor(saturation, scale=1.0)
            img = (img - 0.5) * (1 + (factor - 1) * s) + 0.5
        if vibrance != 0:
            max_rgb = torch.max(img, dim=1, keepdim=True)[0]
            min_rgb = torch.min(img, dim=1, keepdim=True)[0]
            s = (max_rgb - min_rgb) / (max_rgb + 1e-7)
            factor = param_to_factor(vibrance, scale=1.0)
            weight = s
            img = (img - 0.5) * (1 + (factor - 1) * weight) + 0.5
        if temperature != 0:
            factor = param_to_factor(temperature, scale=0.5)
            img[:, 0:1] = img[:, 0:1] * (1 + max(0, factor - 1) * 0.5)
            img[:, 2:3] = img[:, 2:3] * (1 + max(0, 1 - factor) * 0.5)
        if tint != 0:
            factor = param_to_factor(tint, scale=0.3)
            img[:, 1:2] = img[:, 1:2] * factor
        if sharpen != 0:
            sharpen_factor = sharpen / 100.0
            kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=img.dtype, device=img.device)
            kernel = kernel.repeat(3, 1, 1, 1)
            laplacian = F.conv2d(img, kernel, padding=1, groups=3)
            img = img + laplacian * sharpen_factor
        # Noise adjustment
        if noise != 0:
            if noise > 0:
                noise_strength = noise / 100.0
                noise_tensor = torch.randn_like(img) * noise_strength
                img = img + noise_tensor
            else:
                strength = abs(noise) / 100.0
                if strength < 0.2:
                    kernel_size = 3
                    repeat = 1
                elif strength < 0.5:
                    kernel_size = 5
                    repeat = 1
                else:
                    kernel_size = 5
                    repeat = 2
                kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=img.dtype, device=img.device) / (kernel_size * kernel_size)
                kernel = kernel.repeat(C, 1, 1, 1)
                for _ in range(repeat):
                    img = F.conv2d(img, kernel, padding=kernel_size//2, groups=C)
            img = torch.clamp(img, 0, 1)
        if fade != 0:
            fade_factor = fade / 100.0
            img = img * (1 - fade_factor * 0.5) + 0.5 * fade_factor * 0.5

        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 1)
        return (img,)