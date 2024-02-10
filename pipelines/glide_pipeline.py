from PIL import Image
import torch as th
import os

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

has_cuda = th.cuda.is_available()
device = th.device("cpu" if not has_cuda else "cuda")


options = model_and_diffusion_defaults()
options["use_fp16"] = has_cuda
options["timestep_respacing"] = "100"  # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint("base", device))
print("total base parameters", sum(x.numel() for x in model.parameters()))


options_up = model_and_diffusion_defaults_upsampler()
options_up["use_fp16"] = has_cuda
options_up[
    "timestep_respacing"
] = "fast27"  # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint("upsample", device))
print("total upsampler parameters", sum(x.numel() for x in model_up.parameters()))

guidance_scale = 3.0
upsample_temp = 0.997

# options["image_size"] = 128
options_up["image_size"] = 256

# print(options["image_size"])
# print(options_up["image_size"])


# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)


def save_images(batch: th.Tensor, output_folder):
    """Save a batch of images as separate files."""
    # Ensure there is a batch dimension
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)

    # Process images and convert to uint8
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8)

    # Loop through each image in the batch
    for i, img in enumerate(scaled):
        # Convert to CPU and numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # Convert numpy array to image
        img_pil = Image.fromarray(img_np)
        # Save the image to a file
        img_pil.save(f"{output_folder}/image_{i+7}.png")


def glide_pipeline(prompt, num_output_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = num_output_images * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options["text_ctx"]
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * num_output_images + [uncond_tokens] * num_output_images,
            device=device,
        ),
        mask=th.tensor(
            [mask] * num_output_images + [uncond_mask] * num_output_images,
            dtype=th.bool,
            device=device,
        ),
    )

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:num_output_images]
    model.del_cache()

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
        # Text tokens
        tokens=th.tensor([tokens] * num_output_images, device=device),
        mask=th.tensor(
            [mask] * num_output_images,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (
        num_output_images,
        3,
        options_up["image_size"],
        options_up["image_size"],
    )
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:num_output_images]
    model_up.del_cache()

    save_images(up_samples, output_folder)
