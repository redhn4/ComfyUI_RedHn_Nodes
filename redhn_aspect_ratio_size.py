import math

class RedHn_AspectRatioSize_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "宽高比": (["方形1:1", "横版16:9", "横版3:2", "横版4:3", "横版2:1", "竖版9:16", "竖版2:3", "竖版3:4", "竖版1:2"],),
                "长边像素": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "整除因数": (["2", "4", "8", "16", "32", "64", "128", "256", "512"], {"default": "2"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽", "高")
    FUNCTION = "calculate"
    CATEGORY = "RedHn"

    def calculate(self, 宽高比, 长边像素, 整除因数):
        factor = int(整除因数)
        ratio_map = {
            "方形1:1": 1.0,
            "横版16:9": 16/9,
            "横版3:2": 3/2,
            "横版4:3": 4/3,
            "横版2:1": 2.0,
            "竖版9:16": 9/16,
            "竖版2:3": 2/3,
            "竖版3:4": 3/4,
            "竖版1:2": 1/2,
        }
        ratio = ratio_map[宽高比]
        if ratio >= 1.0:  # 横版
            width = 长边像素
            height = width / ratio
        else:  # 竖版
            height = 长边像素
            width = height * ratio

        width = int(round(width))
        height = int(round(height))

        width = int(round(width / factor)) * factor
        height = int(round(height / factor)) * factor

        width = max(width, factor)
        height = max(height, factor)

        return (width, height)


class RedHn_AspectRatioSize_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Aspect Ratio": (["Square 1:1", "Landscape 16:9", "Landscape 3:2", "Landscape 4:3", "Landscape 2:1", "Portrait 9:16", "Portrait 2:3", "Portrait 3:4", "Portrait 1:2"],),
                "Long pixel": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "Divisible by": (["2", "4", "8", "16", "32", "64", "128", "256", "512"], {"default": "2"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "RedHn"

    def calculate(self, Aspect_Ratio, Long_pixel, Divisible_by):
        factor = int(Divisible_by)
        ratio_map = {
            "Square 1:1": 1.0,
            "Landscape 16:9": 16/9,
            "Landscape 3:2": 3/2,
            "Landscape 4:3": 4/3,
            "Landscape 2:1": 2.0,
            "Portrait 9:16": 9/16,
            "Portrait 2:3": 2/3,
            "Portrait 3:4": 3/4,
            "Portrait 1:2": 1/2,
        }
        ratio = ratio_map[Aspect_Ratio]
        if ratio >= 1.0:
            width = Long_pixel
            height = width / ratio
        else:
            height = Long_pixel
            width = height * ratio

        width = int(round(width))
        height = int(round(height))

        width = int(round(width / factor)) * factor
        height = int(round(height / factor)) * factor

        width = max(width, factor)
        height = max(height, factor)

        return (width, height)