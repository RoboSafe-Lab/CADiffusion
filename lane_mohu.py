import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def create_red_mask(red_marked_image_path: str, output_mask_path: str) -> Image.Image:
    """
    根据车道线被红色标记的图像生成二值蒙版。

    参数:
      - red_marked_image_path: 包含红色标记的图像路径
      - output_mask_path: 生成的蒙版保存路径

    返回:
      - mask_pil: 生成的二值蒙版（PIL 图像），车道线区域为白色，其余区域为黑色。
    """
    # 1. 读取图像
    image = cv2.imread(red_marked_image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {red_marked_image_path}")

    # 2. 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 红色在 HSV 中通常分布在两个范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 4. 合并两个 mask
    mask = cv2.bitwise_or(mask1, mask2)

    # 5. 形态学处理，去除噪点并平滑边缘
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. 保存生成的蒙版
    cv2.imwrite(output_mask_path, mask)
    print(f"生成的蒙版已保存至: {output_mask_path}")

    # 7. 转换为 PIL 图像，并转换为 RGB 格式（inpainting 模型要求）
    mask_pil = Image.fromarray(mask).convert("RGB")
    return mask_pil


def process_single_image(
    pipe,
    original_image_path: str,
    red_marked_image_path: str,
    mask_output_path: str,
    output_path: str,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0
) -> None:
    """
    对单张图像进行处理：
      - 生成车道线蒙版（保存至 mask_output_path）
      - 使用 Stable Diffusion Inpainting 修改车道线区域
      - 保存最终结果至 output_path
    """
    # 加载原图，并记录尺寸
    original_pil = Image.open(original_image_path).convert("RGB")
    original_width, original_height = original_pil.size

    # 生成蒙版并调整尺寸匹配原图
    mask_pil = create_red_mask(red_marked_image_path, mask_output_path)
    mask_pil = mask_pil.resize((original_width, original_height))

    # 调用 inpainting 模型进行处理
    result = pipe(
        prompt=prompt,
        image=original_pil,
        mask_image=mask_pil,
        height=original_height,
        width=original_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # 保存处理结果
    result.images[0].save(output_path)
    print(f"最终处理后的图像已保存至: {output_path}")


def process_folder(
    original_folder: str,
    red_marked_folder: str,
    mask_output_folder: str,
    output_folder: str,
    prompt: str,
    model_file: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0
) -> None:
    """
    对文件夹下的所有图片进行处理。假设 original_folder 中的图片和 red_marked_folder 中的图片
    按文件名排序后一一对应。生成的蒙版保存在 mask_output_folder，
    处理后的图像保存在 output_folder。
    """
    # 创建输出文件夹（如果不存在则创建）
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 获取原图与红色标记图像文件列表（支持 jpg, jpeg, png）
    original_files = sorted([
        f for f in os.listdir(original_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    red_marked_files = sorted([
        f for f in os.listdir(red_marked_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(original_files) != len(red_marked_files):
        print("警告：原图和红色标记图像数量不一致！将以较少的文件数为准。")
    count = min(len(original_files), len(red_marked_files))
    print(f"共找到 {count} 对图像，开始处理...")

    # 加载 Stable Diffusion Inpainting 模型（仅加载一次）
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        pretrained_model_link_or_path=model_file,
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None
    ).to("cuda")

    # 尝试启用 CPU offload，降低 GPU 内存占用
    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print("无法启用 CPU offload:", e)

    # 按顺序遍历所有图像对
    for i in range(count):
        orig_filename = original_files[i]
        red_filename = red_marked_files[i]

        original_path = os.path.join(original_folder, orig_filename)
        red_marked_path = os.path.join(red_marked_folder, red_filename)
        mask_output_path = os.path.join(mask_output_folder, f"mask_{orig_filename}")
        output_path = os.path.join(output_folder, f"processed_{orig_filename}")

        print(f"\n处理第 {i+1} 张图像：")
        print(f"原图: {original_path}")
        print(f"红色标记图像: {red_marked_path}")

        process_single_image(
            pipe,
            original_image_path=original_path,
            red_marked_image_path=red_marked_path,
            mask_output_path=mask_output_path,
            output_path=output_path,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )


if __name__ == "__main__":
    # 请根据实际情况修改以下文件夹路径
    original_folder = r"D:\Work\Course\Project\YOLOP\inference\images"
    red_marked_folder = r"D:\Work\Course\Project\YOLOP\inference\output"
    mask_output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\masks"
    output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\mosun"
    model_file = r""

    prompt = "Blur the lane lines while keeping the rest of the scene intact"

    process_folder(
        original_folder=original_folder,
        red_marked_folder=red_marked_folder,
        mask_output_folder=mask_output_folder,
        output_folder=output_folder,
        prompt=prompt,
        model_file=model_file
    )
