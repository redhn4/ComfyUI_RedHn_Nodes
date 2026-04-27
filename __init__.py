from .RedHn_Quick_Resolution import RedHn_Quick_Resolution_CN, RedHn_Quick_Resolution_EN
from .redhn_switch_ab import RedHn_SwitchAB_CN, RedHn_SwitchAB_EN
from .redhn_image_adjust import RedHn_ImageAdjust_CN, RedHn_ImageAdjust_EN
from .redhn_batch_images import RedHn_BatchImages_CN, RedHn_BatchImages_EN
from .redhn_batch_images_pro import RedHn_BatchImagesPro_CN, RedHn_BatchImagesPro_EN
from .redhn_hsl_mixer import RedHn_HSL_Mixer_CN, RedHn_HSL_Mixer_EN
from .redhn_quick_mv import NODE_CLASS_MAPPINGS as MV_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MV_DISPLAY

NODE_CLASS_MAPPINGS = {
    "RedHn快捷分辨率": RedHn_Quick_Resolution_CN,
    "RedHn Quick Resolution": RedHn_Quick_Resolution_EN,
    "RedHn切换AB": RedHn_SwitchAB_CN,
    "RedHn Switch AB": RedHn_SwitchAB_EN,
    "RedHn图像调整": RedHn_ImageAdjust_CN,
    "RedHn Image Adjust": RedHn_ImageAdjust_EN,
    "RedHn图像批处理": RedHn_BatchImages_CN,                 # 基础版
    "RedHn Batch Images": RedHn_BatchImages_EN,
    "RedHn图像批处理Pro": RedHn_BatchImagesPro_CN,           # Pro版
    "RedHn Batch Images Pro": RedHn_BatchImagesPro_EN,
    "RedHn混色器HSL": RedHn_HSL_Mixer_CN,
    "RedHn HSL Mixer": RedHn_HSL_Mixer_EN,
    **MV_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RedHn快捷分辨率": "RedHn快捷分辨率",
    "RedHn Quick Resolution": "RedHn Quick Resolution",
    "RedHn切换AB": "RedHn切换AB",
    "RedHn Switch AB": "RedHn Switch AB",
    "RedHn图像调整": "RedHn图像调整",
    "RedHn Image Adjust": "RedHn Image Adjust",
    "RedHn图像批处理": "RedHn图像批处理",
    "RedHn Batch Images": "RedHn Batch Images",
    "RedHn图像批处理Pro": "RedHn图像批处理Pro",
    "RedHn Batch Images Pro": "RedHn Batch Images Pro",
    "RedHn混色器HSL": "RedHn混色器HSL",
    "RedHn HSL Mixer": "RedHn HSL Mixer",
    **MV_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']