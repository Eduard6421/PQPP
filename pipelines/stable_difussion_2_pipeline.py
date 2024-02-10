import os
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


def stable_difussion_2_pipeline(prompt, num_output_images, output_folder):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prompt = [prompt] * num_output_images
    images = pipe(prompt)

    print(len(images[0]))

    for i in range(num_output_images):
        images[0][i].save(f"{output_folder}/astronaut_rides_horse_{i+1}.png")
