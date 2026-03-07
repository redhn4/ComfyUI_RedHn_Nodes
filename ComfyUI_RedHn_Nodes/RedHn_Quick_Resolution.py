import torch

# 中文预设保持不变
RESOLUTION_PRESETS_CN = [
    "自定义分辨率",
    "4K_横版16:9_3840x2160",
    "4K_竖版9:16_2160x3840", 
    "2K_横版16:9_2560x1440",
    "2K_竖版9:16_1440x2560",
    "1080p_横版16:9_1920x1080",
    "1080p_竖版9:16_1080x1920",
    "720p_横版16:9_1280x720",
    "720p_竖版9:16_720x1280",
    "iPhone_竖版19.5:9_1208x2624",
    "Android_竖版20:9_1080x2400",
    "横版16:9高_2048x1152",
    "横版16:9高_1920x1080",
    "横版16:9中_1664x928",
    "横版16:9中_1536x864",
    "横版16:9低_1344x768",
    "横版16:9低_1024x576",
    "横版4:3高_2048x1536",
    "横版4:3中_1536x1152",
    "横版4:3中_1472x1104",
    "横版4:3低_1024x768",
    "横版3:2高_1920x1280",
    "横版3:2中_1584x1056",
    "横版3:2中_1536x1024",
    "横版3:2低_1280x864",
    "竖版9:16高_1152x2048",
    "竖版9:16高_1080x1920",
    "竖版9:16中_928x1664",
    "竖版9:16中_864x1536",
    "竖版9:16低_768x1344",
    "竖版9:16低_576x1024",
    "竖版3:4高_1536x2048",
    "竖版3:4中_1152x1536",
    "竖版3:4中_1104x1472",
    "竖版3:4低_768x1024",
    "竖版2:3高_1280x1920",
    "竖版2:3中_1056x1584",
    "竖版2:3中_1024x1536",
    "竖版2:3低_864x1280",
    "方1:1高_4096x4096",
    "方1:1高_3840x3840",
    "方1:1高_2560x2560",
    "方1:1高_2160x2160",
    "方1:1高_2048x2048",
    "方1:1中_1536x1536",
    "方1:1中_1328x1328",
    "方1:1中_1024x1024",
    "方1:1低_768x768",
    "方1:1低_512x512"
]

# 简化英文预设 - 去掉所有横竖标识，只保留比例和尺寸
RESOLUTION_PRESETS_EN = [
    "Custom Resolution",
    # 标准分辨率
    "4K_16:9_3840x2160",
    "4K_9:16_2160x3840",
    "2K_16:9_2560x1440",
    "2K_9:16_1440x2560",
    "1080p_16:9_1920x1080",
    "1080p_9:16_1080x1920",
    "720p_16:9_1280x720",
    "720p_9:16_720x1280",
    # 手机分辨率
    "iPhone_19.5:9_1208x2624",
    "Android_20:9_1080x2400",
    # 16:9 比例
    "16:9_High_2048x1152",
    "16:9_High_1920x1080",
    "16:9_Mid_1664x928",
    "16:9_Mid_1536x864",
    "16:9_Low_1344x768",
    "16:9_Low_1024x576",
    # 4:3 比例
    "4:3_High_2048x1536",
    "4:3_Mid_1536x1152",
    "4:3_Mid_1472x1104",
    "4:3_Low_1024x768",
    # 3:2 比例
    "3:2_High_1920x1280",
    "3:2_Mid_1584x1056",
    "3:2_Mid_1536x1024",
    "3:2_Low_1280x864",
    # 9:16 比例
    "9:16_High_1152x2048",
    "9:16_High_1080x1920",
    "9:16_Mid_928x1664",
    "9:16_Mid_864x1536",
    "9:16_Low_768x1344",
    "9:16_Low_576x1024",
    # 3:4 比例
    "3:4_High_1536x2048",
    "3:4_Mid_1152x1536",
    "3:4_Mid_1104x1472",
    "3:4_Low_768x1024",
    # 2:3 比例
    "2:3_High_1280x1920",
    "2:3_Mid_1056x1584",
    "2:3_Mid_1024x1536",
    "2:3_Low_864x1280",
    # 1:1 方形
    "1:1_High_4096x4096",
    "1:1_High_3840x3840",
    "1:1_High_2560x2560",
    "1:1_High_2160x2160",
    "1:1_High_2048x2048",
    "1:1_Mid_1536x1536",
    "1:1_Mid_1328x1328",
    "1:1_Mid_1024x1024",
    "1:1_Low_768x768",
    "1:1_Low_512x512"
]

def parse_resolution(resolution_str):
    """解析分辨率字符串，返回宽高"""
    # 从字符串中提取分辨率部分（最后一个下划线后的内容）
    if '_' in resolution_str:
        resolution_part = resolution_str.split('_')[-1]
    else:
        resolution_part = resolution_str
    
    # 解析宽高
    if 'x' in resolution_part:
        width_str, height_str = resolution_part.split('x')
        try:
            width = int(width_str)
            height = int(height_str)
            return width, height
        except ValueError:
            return 1024, 1024
    return 1024, 1024

class RedHn_Quick_Resolution_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "选择分辨率": (RESOLUTION_PRESETS_CN, {"default": "自定义分辨率"}),
                "自定义宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "自定义高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "长宽比切换": ("BOOLEAN", {"default": False, "label_on": "开", "label_off": "关"}),
                "批量大小": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("Latent", "宽", "高", "尺寸信息")
    FUNCTION = "generate_latent"
    CATEGORY = "RedHn"

    def generate_latent(self, 选择分辨率, 自定义宽度, 自定义高度, 长宽比切换, 批量大小):
        # 解析分辨率
        if 选择分辨率 == "自定义分辨率":
            width = 自定义宽度
            height = 自定义高度
        else:
            width, height = parse_resolution(选择分辨率)
        
        # 应用长宽比切换
        if 长宽比切换:
            width, height = height, width
        
        # 创建潜在空间
        latent = torch.zeros([批量大小, 4, height // 8, width // 8])
        
        # 生成尺寸信息
        size_info = f"{width}x{height} (批量: {批量大小})"
        
        return ({"samples": latent}, width, height, size_info)

class RedHn_Quick_Resolution_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Resolution_Select": (RESOLUTION_PRESETS_EN, {"default": "Custom Resolution"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "swap_aspect_ratio": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),
                "Batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("Latent", "width", "height", "size_info")
    FUNCTION = "generate_latent"
    CATEGORY = "RedHn"

    def generate_latent(self, Resolution_Select, custom_width, custom_height, swap_aspect_ratio, Batch_size):
        # 解析分辨率
        if Resolution_Select == "Custom Resolution":
            width = custom_width
            height = custom_height
        else:
            width, height = parse_resolution(Resolution_Select)
        
        # 应用长宽比切换
        if swap_aspect_ratio:
            width, height = height, width
        
        # 创建潜在空间
        latent = torch.zeros([Batch_size, 4, height // 8, width // 8])
        
        # 生成尺寸信息
        size_info = f"{width}x{height} (batch: {Batch_size})"
        
        return ({"samples": latent}, width, height, size_info)