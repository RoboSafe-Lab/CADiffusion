import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# 禁用不必要的警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.backends.cudnn.benchmark = True

# 全局加载 BLIP 图像描述模型，用于生成详细提示词
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def generate_lane_edges(input_path: str, output_path: str) -> np.ndarray:
    """
    生成主视角车道线边缘图
    """
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_path, edge_rgb)
    return edge_rgb


def generate_detailed_prompt(image_path: str, rainy_keywords: str) -> str:
    """
    利用 BLIP 模型将图像转换为详细的描述性提示词，并附加雨天关键词
    这里使用了较大的 beam size 以期获得更详细的描述。
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda")
    out = caption_model.generate(**inputs, num_beams=5, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    # 组合详细描述与雨天关键词
    prompt = f"{caption}, {rainy_keywords}"
    return prompt


def cleanup_prompt_model():
    """
    释放 BLIP 模型相关资源
    """
    global caption_model, processor
    caption_model.cpu()
    del caption_model
    del processor
    torch.cuda.empty_cache()


class SingleViewGenerator:
    def __init__(self, model_path: str):
        """初始化生成器，使用单个 pipeline 生成单视角图像"""
        self.model_path = model_path
        self.pipe = self._init_pipeline("lllyasviel/sd-controlnet-canny")

    def _init_pipeline(self, controlnet_path: str):
        """初始化生成管道"""
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        pipe = StableDiffusionControlNetPipeline.from_single_file(
            pretrained_model_link_or_path=self.model_path,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
            local_files_only=True
        ).to("cuda")

        # 尝试启用 CPU offload，减少 GPU 内存占用
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print("无法启用 CPU offload:", e)
        return pipe

    def generate_view(
            self,
            edge_image_path: str,
            prompt: str,
            output_path: str,
            **generation_args
    ) -> str:
        """
        根据边缘图和提示词生成单视角场景
        """
        default_args = {
            "num_inference_steps": 50,
            "guidance_scale": 3,
        }
        default_args.update(generation_args)
        seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)
        default_args["generator"] = generator

        # 加载并调整边缘图（调整为 1280x720）
        edge_image = Image.open(edge_image_path).convert("RGB")
        edge_image = edge_image.resize((1280, 720))

        # 生成图像
        result = self.pipe(
            prompt=prompt,
            image=edge_image,
            **default_args
        )
        result.images[0].save(output_path)
        torch.cuda.empty_cache()
        return output_path


def process_folder(
        original_folder: str,
        edge_output_folder: str,
        prompt_output_folder: str,
        output_folder: str,
        model_path: str,
        rainy_keywords: str
):
    """
    1. 遍历输入文件夹，生成每张图片对应的详细提示词，并保存到 prompt_output_folder
    2. 所有提示词生成完成后，再遍历图片生成边缘图，调用生成器根据提示词生成雨天效果图
    """
    os.makedirs(edge_output_folder, exist_ok=True)
    os.makedirs(prompt_output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 阶段1：生成提示词并保存到文件
    prompts_dict = {}  # 用于保存 {文件名: 提示词} 的对应关系
    for file_name in os.listdir(original_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(original_folder, file_name)
            prompt = generate_detailed_prompt(input_path, rainy_keywords)
            prompts_dict[file_name] = prompt
            # 将提示词保存到文本文件
            prompt_file = os.path.splitext(file_name)[0] + ".txt"
            prompt_path = os.path.join(prompt_output_folder, prompt_file)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"生成提示词: {file_name} -> {prompt_file}")

    # 提示词生成完毕后，释放提示词模型资源
    cleanup_prompt_model()

    # 阶段2：生成雨天效果图
    generator = SingleViewGenerator(model_path)
    for file_name, prompt in prompts_dict.items():
        input_path = os.path.join(original_folder, file_name)
        # 生成边缘图
        edge_file = os.path.splitext(file_name)[0] + "_edge.png"
        edge_path = os.path.join(edge_output_folder, edge_file)
        generate_lane_edges(input_path, edge_path)

        # 生成雨天效果图
        output_file = os.path.splitext(file_name)[0] + "_rainy.png"
        output_path = os.path.join(output_folder, output_file)
        generator.generate_view(
            edge_image_path=edge_path,
            prompt=prompt,
            output_path=output_path
        )
        print(f"处理完成: {file_name} -> {output_file}")


if __name__ == "__main__":
    # 输入图片所在文件夹
    original_folder = r"D:\Work\Course\Project\YOLOP\inference\images"
    # 边缘图保存文件夹
    edge_output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\edges"
    # 提示词保存文件夹（将每张图片对应的详细提示词保存为文本文件）
    prompt_output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\prompts"
    # 最终生成雨天场景图像保存文件夹
    output_folder = r"D:\Work\Course\Project\DenoisingDiffusionProbabilityModel-ddpm-\rainy"
    # Stable Diffusion 模型文件路径（.safetensors 文件）
    model_path = r""
    # 雨天关键词，将自动生成的描述与该关键词组合，确保描述足够详细
    rainy_keywords = (
        "transformed by rainy weather, featuring wet asphalt, subtle rain droplets, and a gloomy, low-saturation atmosphere"
    )

    process_folder(original_folder, edge_output_folder, prompt_output_folder, output_folder, model_path, rainy_keywords)
