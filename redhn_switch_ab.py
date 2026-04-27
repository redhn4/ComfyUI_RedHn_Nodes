class RedHn_SwitchAB_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "切换": ("BOOLEAN", {"default": False, "label_on": "开", "label_off": "关"}),
                "A": ("*", {"forceInput": True}),
                "B": ("*", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("A输出", "B输出")
    FUNCTION = "switch"
    CATEGORY = "RedHn"

    def switch(self, 切换, A=None, B=None):
        if 切换:
            return (B, A)
        else:
            return (A, B)


class RedHn_SwitchAB_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                "A": ("*", {"forceInput": True}),
                "B": ("*", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("A_out", "B_out")
    FUNCTION = "switch"
    CATEGORY = "RedHn"

    def switch(self, switch, A=None, B=None):
        if switch:
            return (B, A)
        else:
            return (A, B)