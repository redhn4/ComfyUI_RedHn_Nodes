# RedHn Nodes for ComfyUI

**RedHn Nodes** 是一套为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 开发的自定义节点集，提供图像调整、批量处理、裁剪拆分、保存等多种实用工具。所有节点均支持中英文双语，命名统一以 **RedHn** 开头，便于识别和搜索。

---

## ✨ 节点列表

| 中文名称 | 英文名称 | 功能简述 |
|---------|---------|---------|
| RedHn快捷分辨率 | RedHn Quick Resolution | 预设常用分辨率（4K、1080p、720p、Flux、SDXL 等），支持自定义宽高和长宽比切换 |
| RedHn切换AB | RedHn Switch AB | 交换两个输入（A↔B），支持任意数据类型，带开关控制 |
| RedHn图像调整 | RedHn Image Adjust | 类似 Photoshop 的图像调整：曝光、亮度、对比度、高光、阴影、饱和度、自然饱和度、色温、色调、锐化、褪色、噪点（正加噪/负降噪） |
| RedHn图像批处理 | RedHn Batch Images | 从多个文件夹批量加载图像，输出图像批次（IMAGE）和图像列表（LIST），支持排序、数量限制 |
| RedHn图像批处理Pro | RedHn Batch Images Pro | 增强版图像批处理，支持“每文件夹”和“全局共计”两种数量模式，适合更复杂的批量场景 |
| RedHn混色器HSL | RedHn HSL Mixer | 类似 Camera Raw 的 HSL 调色：分别调节 8 种颜色的色相、饱和度、明亮度（红、橙、黄、绿、青、蓝、紫、洋红） |
| RedHn九宫格拆分 | RedHn 9Grid Splitter | 将输入图像等分为 3×3 的九宫格子图，支持“裁切前收缩”和“裁切后收缩”以去除白边 |
| RedHn宫格拆分 | RedHn Grid Splitter | 自定义行列数（1~10）的宫格拆分，输出子图列表，用于分镜图等场景 |
| RedHn保存图片 | RedHn Save Image | 保存图像为多种格式（JPG、PNG、WEBP、BMP、TGA），支持自动序号防覆盖、时间戳文件名、PNG 嵌入工作流 |
| RedHn像素收缩 | RedHn Pixel Shrink | 将图像四周向内均匀收缩指定百分比（0~100），用于去边框 |
| RedHn扩展边框 | RedHn Expand Border | 为图像添加边框，支持两种模式：像素值（1-512）或短边比（1-50%），可启用圆角并控制圆角值，支持批处理 |
| RedHn宽高比分辨率 | RedHn Aspect Ratio Size | 根据宽高比（方形、横版16:9、3:2、4:3、2:1，竖版9:16、2:3、3:4、1:2）和长边像素，计算宽高，并自动对齐整除因数 |

> 注：`redhn_quick_mv` 节点为第三方扩展，本套件保留其兼容性。

---

## 🚀 安装

1. 进入 ComfyUI 的 `custom_nodes` 目录。
2. 克隆本仓库：
   ```bash
   git clone https://github.com/yourusername/ComfyUI_RedHn_Nodes.git