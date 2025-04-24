import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler


def create_vehicle_mask(red_marked_image_path: str, output_mask_path: str,
                        final_axis_x: int, final_axis_y: int,
                        original_size: tuple = None) -> Image.Image:
    """
    优化版车辆轮廓蒙版生成函数，采用新的车辆后视轮廓模板，
    并根据距离图像底部的位置动态调整模板大小。
    """
    # 读取并处理图像
    image = cv2.imread(red_marked_image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {red_marked_image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色检测
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 计算目标中心：图像中下部（目标点定义为宽度中心，75%高度位置）
    height, width = mask.shape
    target_center = (width // 2, int(height * 0.75))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = target_center
    if contours:
        min_dist = float('inf')
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.hypot(cx - target_center[0], cy - target_center[1])
                if dist < min_dist:
                    min_dist = dist
                    best_center = (cx, cy)

    # 计算车道角度（利用霍夫直线检测获取平均角度）
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    lane_angle = np.mean(angles) if angles else 0.0

    # 新的车辆后视模板轮廓（左右对称，上窄下宽，更符合车辆后视）
    vehicle_polygon = np.array([
        [-0.6, 0.0],   # 左下
        [-0.6, -0.5],   # 左侧中部
        [-0.5, -0.8],   # 左上部分
        [-0.2, -0.9],   # 左上尖端
        [0.2, -0.9],    # 右上尖端
        [0.5, -0.8],    # 右上部分
        [0.6, -0.5],    # 右侧中部
        [0.6, 0.0]     # 右下
    ], dtype=np.float32)

    # 动态缩放：根据 best_center 距离图像底部的距离决定模板大小
    multiplier = (0.1+(best_center[1])/ (height))  # 当 best_center 靠近底部时乘数较大
    # 以 final_axis_x 和 final_axis_y 为基础尺寸，计算缩放因子
    min_x, max_x = vehicle_polygon[:, 0].min(), vehicle_polygon[:, 0].max()
    min_y, max_y = vehicle_polygon[:, 1].min(), vehicle_polygon[:, 1].max()
    scale_x = (2 * final_axis_x) / (max_x - min_x)
    scale_y = (2 * final_axis_y) / (max_y - min_y)
    base_scale = min(scale_x, scale_y)
    scale = base_scale * multiplier
    vehicle_polygon_scaled = vehicle_polygon * scale

    # 旋转变换，使模板与车道角度对齐
    theta = np.radians(-lane_angle/20)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
    vehicle_polygon_rotated = vehicle_polygon_scaled.dot(rot_mat.T) + best_center

    # 在空白蒙版上绘制车辆轮廓
    mask_result = np.zeros((height, width), dtype=np.uint8)
    pts = np.int32(vehicle_polygon_rotated).reshape((-1, 1, 2))
    cv2.fillPoly(mask_result, [pts], 255)
    mask_result = cv2.dilate(mask_result, kernel, iterations=2)

    # 保存蒙版并转换为 PIL 格式（RGB模式）
    cv2.imwrite(output_mask_path, mask_result)
    mask_pil = Image.fromarray(mask_result).convert("RGB")
    if original_size and mask_pil.size != original_size:
        mask_pil = mask_pil.resize(original_size)
    return mask_pil


def process_single_image(
        pipe,
        original_image_path: str,
        red_marked_image_path: str,
        mask_output_path: str,
        output_path: str,
        prompt: str,
        final_axis_x: int,
        final_axis_y: int,
        num_inference_steps: int = 75,
        guidance_scale: float = 11.0
) -> None:
    # 加载原图并获取尺寸
    original_pil = Image.open(original_image_path).convert("RGB")
    original_size = original_pil.size

    # 生成车辆蒙版
    mask_pil = create_vehicle_mask(
        red_marked_image_path, mask_output_path,
        final_axis_x, final_axis_y, original_size
    )

    # 构建强化提示词和负面提示词
    enhanced_prompt = (
        "car rear view" + prompt
    )
    # 调用 inpainting 模型进行修复
    result = pipe(
        prompt=enhanced_prompt,
        image=original_pil,
        mask_image=mask_pil,
        height=original_size[1],
        width=original_size[0],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(2511),
        cross_attention_kwargs={"scale": 1}
    )

    # 保存处理后的图像
    result.images[0].save(output_path)
    print(f"Saved result to {output_path}")


def process_folder(
        original_folder: str,
        red_marked_folder: str,
        mask_output_folder: str,
        output_folder: str,
        prompt: str,
        model_file: str,
        final_axis_x: int,
        final_axis_y: int,
        num_inference_steps: int = 75,
        guidance_scale: float = 15.0
) -> None:
    # 创建输出目录（如果不存在则创建）
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 加载 inpainting 模型
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        model_file,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
        variant="fp16",
        custom_pipeline="lpw_stable_diffusion"
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print("无法启用 CPU offload:", e)

    # 获取原图和红色标记图像列表
    orig_files = sorted([f for f in os.listdir(original_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    red_files = sorted([f for f in os.listdir(red_marked_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 对每一对图像进行处理
    for i in range(min(len(orig_files), len(red_files))):
        orig_path = os.path.join(original_folder, orig_files[i])
        red_path = os.path.join(red_marked_folder, red_files[i])
        mask_path = os.path.join(mask_output_folder, f"mask_{orig_files[i]}")
        out_path = os.path.join(output_folder, f"processed_{orig_files[i]}")

        print(f"\nProcessing {i + 1}/{len(orig_files)}:")
        print(f"Original: {orig_path}")
        print(f"Red marked: {red_path}")

        process_single_image(
            pipe, orig_path, red_path, mask_path, out_path,
            prompt, final_axis_x, final_axis_y,
            num_inference_steps, guidance_scale
        )


if __name__ == "__main__":
    # 配置参数（请根据实际情况修改路径和参数）
    original_folder = r".\images"
    red_marked_folder = r".\output"
    mask_output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\masks_car"
    output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\car"
    model_file = r""

    prompt = (
        "'grey car',driving forward on the 'road',"
    )

    process_folder(
        original_folder=original_folder,
        red_marked_folder=red_marked_folder,
        mask_output_folder=mask_output_folder,
        output_folder=output_folder,
        prompt=prompt,
        model_file=model_file,
        final_axis_x=800,  # 根据实际车辆尺寸调整
        final_axis_y=200,
        num_inference_steps=100,
        guidance_scale=3.0
    )
