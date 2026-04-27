import torch

class RedHn_HSL_Mixer_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "启用": ("BOOLEAN", {"default": True, "label_on": "开", "label_off": "关"}),
                "色相_红色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_橙色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_黄色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_浅绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_蓝色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_紫色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "色相_洋红": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_红色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_橙色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_黄色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_浅绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_蓝色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_紫色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "饱和度_洋红": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_红色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_橙色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_黄色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_浅绿色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_蓝色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_紫色": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "明亮度_洋红": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "mix"
    CATEGORY = "RedHn"

    def rgb_to_hsv(self, img):
        # img shape: [B, C, H, W]
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        max_val, _ = torch.max(img, dim=1, keepdim=True)
        min_val, _ = torch.min(img, dim=1, keepdim=True)
        delta = max_val - min_val
        h = torch.zeros_like(max_val)
        mask = delta != 0
        r_eq = (r == max_val) & mask
        g_eq = (g == max_val) & mask
        b_eq = (b == max_val) & mask
        h[r_eq] = ((g[r_eq] - b[r_eq]) / delta[r_eq]) % 6
        h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 2
        h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 4
        h = h / 6.0
        s = torch.where(max_val == 0, torch.zeros_like(max_val), delta / max_val)
        v = max_val
        return h, s, v

    def hsv_to_rgb(self, h, s, v):
        # h, s, v shape: [B, 1, H, W]
        h = h * 6.0
        i = torch.floor(h).long()
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i_mod = i % 6
        r, g, b = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)
        
        m0, m1, m2, m3, m4, m5 = (i_mod == 0), (i_mod == 1), (i_mod == 2), (i_mod == 3), (i_mod == 4), (i_mod == 5)
        
        r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
        
        return torch.cat([r, g, b], dim=1) # [B, 3, H, W]

    def mix(self, 图像, 启用, **kwargs):
        if not 启用:
            return (图像,)

        # 核心逻辑：强制将输入视为 [N, H, W, C] 处理
        # 如果是 5D (B, F, H, W, C)，合并前两维
        orig_shape = 图像.shape
        if 图像.dim() == 5:
            B, F, H, W, C = orig_shape
            img_in = 图像.reshape(-1, H, W, C)
        else:
            img_in = 图像

        # 转换到 [N, C, H, W] 进行颜色计算
        img = img_in.permute(0, 3, 1, 2).contiguous()
        h, s, v = self.rgb_to_hsv(img)

        centers_deg = {"红色":0, "橙色":30, "黄色":60, "绿色":120, "浅绿色":180, "蓝色":240, "紫色":300, "洋红":330}
        bandwidth = 0.1

        hue_shift = torch.zeros_like(h)
        sat_scale = torch.ones_like(s)
        val_scale = torch.ones_like(v)

        for color, center_deg in centers_deg.items():
            center = center_deg / 360.0
            dist = torch.abs(h - center)
            dist = torch.min(dist, 1.0 - dist)
            weight = torch.exp(-(dist ** 2) / (2 * bandwidth ** 2))

            hue_val = kwargs.get(f"色相_{color}", 0) / 100.0 * 0.0833
            sat_val = kwargs.get(f"饱和度_{color}", 0) / 100.0
            light_val = kwargs.get(f"明亮度_{color}", 0) / 100.0

            hue_shift += weight * hue_val
            sat_scale *= (1.0 + weight * sat_val)
            val_scale *= (1.0 + weight * light_val)

        h_new = (h + hue_shift) % 1.0
        s_new = torch.clamp(s * sat_scale, 0.0, 1.0)
        v_new = torch.clamp(v * val_scale, 0.0, 1.0)

        rgb = self.hsv_to_rgb(h_new, s_new, v_new)
        # 此时 rgb 确定是 4 维 [N, 3, H, W]
        out = rgb.permute(0, 2, 3, 1).contiguous()

        # 如果原图是 5 维，恢复维度
        if 图像.dim() == 5:
            out = out.reshape(B, F, H, W, C)

        return (out,)

class RedHn_HSL_Mixer_EN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
                "Hue_Red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Orange": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Yellow": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Cyan": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Purple": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Hue_Magenta": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Orange": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Yellow": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Cyan": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Purple": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Sat_Magenta": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Orange": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Yellow": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Cyan": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Purple": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "Light_Magenta": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mix"
    CATEGORY = "RedHn"

    def rgb_to_hsv(self, img):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        max_val, _ = torch.max(img, dim=1, keepdim=True)
        min_val, _ = torch.min(img, dim=1, keepdim=True)
        delta = max_val - min_val
        h = torch.zeros_like(max_val)
        mask = delta != 0
        r_eq = (r == max_val) & mask
        g_eq = (g == max_val) & mask
        b_eq = (b == max_val) & mask
        h[r_eq] = ((g[r_eq] - b[r_eq]) / delta[r_eq]) % 6
        h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 2
        h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 4
        h = h / 6.0
        s = torch.where(max_val == 0, torch.zeros_like(max_val), delta / max_val)
        v = max_val
        return h, s, v

    def hsv_to_rgb(self, h, s, v):
        h = h * 6.0
        i = torch.floor(h).long()
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i_mod = i % 6
        r, g, b = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)
        m0, m1, m2, m3, m4, m5 = (i_mod == 0), (i_mod == 1), (i_mod == 2), (i_mod == 3), (i_mod == 4), (i_mod == 5)
        r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
        return torch.cat([r, g, b], dim=1)

    def mix(self, image, enable, **kwargs):
        if not enable:
            return (image,)

        orig_shape = image.shape
        if image.dim() == 5:
            B, F, H, W, C = orig_shape
            img_in = image.reshape(-1, H, W, C)
        else:
            img_in = image

        img = img_in.permute(0, 3, 1, 2).contiguous()
        h, s, v = self.rgb_to_hsv(img)

        centers_deg = {"Red":0, "Orange":30, "Yellow":60, "Green":120, "Cyan":180, "Blue":240, "Purple":300, "Magenta":330}
        bandwidth = 0.1

        hue_shift = torch.zeros_like(h)
        sat_scale = torch.ones_like(s)
        val_scale = torch.ones_like(v)

        for color, center_deg in centers_deg.items():
            center = center_deg / 360.0
            dist = torch.abs(h - center)
            dist = torch.min(dist, 1.0 - dist)
            weight = torch.exp(-(dist ** 2) / (2 * bandwidth ** 2))

            hue_val = kwargs.get(f"Hue_{color}", 0) / 100.0 * 0.0833
            sat_val = kwargs.get(f"Sat_{color}", 0) / 100.0
            light_val = kwargs.get(f"Light_{color}", 0) / 100.0

            hue_shift += weight * hue_val
            sat_scale *= (1.0 + weight * sat_val)
            val_scale *= (1.0 + weight * light_val)

        h_new = (h + hue_shift) % 1.0
        s_new = torch.clamp(s * sat_scale, 0.0, 1.0)
        v_new = torch.clamp(v * val_scale, 0.0, 1.0)

        rgb = self.hsv_to_rgb(h_new, s_new, v_new)
        out = rgb.permute(0, 2, 3, 1).contiguous()

        if image.dim() == 5:
            out = out.reshape(B, F, H, W, C)

        return (out,)