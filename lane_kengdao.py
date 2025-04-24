import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def inpaint_image(
    original_image_path: str,
    mask_image_path: str,
    model_file: str,
    output_path: str,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    device: str = "cuda"
) -> None:
    """
    对单张图像进行 inpainting 处理。

    参数:
      - original_image_path: 原始图像路径
      - mask_image_path: 二值蒙版图像路径（白色区域为待修复区域）
      - model_file: Stable Diffusion inpainting 模型文件或路径
      - output_path: 生成结果的保存路径
      - prompt: 生成提示词
      - num_inference_steps: 推理步数（默认 50）
      - guidance_scale: 生成引导强度（默认 9.0）
      - device: 运行设备，如 "cuda" 或 "cpu"
    """
    # 载入图像和蒙版
    original = Image.open(original_image_path).convert("RGB")
    mask = Image.open(mask_image_path).convert("RGB").resize(original.size)

    # 加载 inpainting 管道
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        pretrained_model_link_or_path=model_file,
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None
    ).to(device)

    # 尝试开启 CPU offload 以减小 GPU 内存占用
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass

    # 运行 inpainting
    result = pipe(
        prompt=prompt,
        image=original,
        mask_image=mask,
        height=original.height,
        width=original.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"生成的图像已保存至: {output_path}")


# --- 直接在此处设置参数并运行 ---
if __name__ == "__main__":
    # 设置输入输出路径及模型
    original_image_path = r"images/1.png"
    mask_image_path = r"masks/mask.png"
    model_file = r""
    output_path = r"car/inpainted.png"

    # 设置生成提示词、推理步数与引导强度
    prompt = "a realistic car parked on a street"
    num_inference_steps = 50
    guidance_scale = 9.0
    device = "cuda"  # 或 "cpu"

    inpaint_image(
        original_image_path=original_image_path,
        mask_image_path=mask_image_path,
        model_file=model_file,
        output_path=output_path,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        device=device
    )
