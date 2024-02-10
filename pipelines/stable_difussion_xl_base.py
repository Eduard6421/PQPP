import os
import torch
from diffusers import DiffusionPipeline

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base.to("cuda")

base.enable_model_cpu_offload()


def stable_difussion_xl_base_pipeline(prompt, num_output_images, output_folder):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prompt = [prompt] * num_output_images
    # run both experts
    base_images = base(
        prompt=prompt,
    ).images

    for i in range(num_output_images):
        base_images[i].save(f"{output_folder}/image_{i+1}.png")
