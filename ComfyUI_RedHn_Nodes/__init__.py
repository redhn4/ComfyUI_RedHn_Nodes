from .RedHn_Quick_Resolution import RedHn_Quick_Resolution_CN, RedHn_Quick_Resolution_EN

NODE_CLASS_MAPPINGS = {
    "RedHn快捷分辨率": RedHn_Quick_Resolution_CN,
    "RedHn Quick Resolution": RedHn_Quick_Resolution_EN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RedHn快捷分辨率": "RedHn快捷分辨率",
    "RedHn Quick Resolution": "RedHn Quick Resolution",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']