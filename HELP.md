Stable Diffusion XL
Open In Colab
Open In Studio Lab
Stable Diffusion XL (SDXL) is a powerful text-to-image generation model that iterates on the previous Stable Diffusion models in three key ways:

the UNet is 3x larger and SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly increase the number of parameters
introduces size and crop-conditioning to preserve training data from being discarded and gain more control over how a generated image should be cropped
introduces a two-stage model process; the base model (can also be run as a standalone model) generates an image as an input to the refiner model which adds additional high-quality details
This guide will show you how to use SDXL for text-to-image, image-to-image, and inpainting.

Before you begin, make sure you have the following libraries installed:

Copied
# uncomment to install the necessary libraries in Colab
#!pip install -q diffusers transformers accelerate invisible-watermark>=0.2.0
We recommend installing the invisible-watermark library to help identify images that are generated. If the invisible-watermark library is installed, it is used by default. To disable the watermarker:

Copied
pipeline = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
Load model checkpoints
Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the from_pretrained() method:

Copied
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
You can also use the from_single_file() method to load a model checkpoint stored in a single file format (.ckpt or .safetensors) from the Hub or locally:

Copied
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors", torch_dtype=torch.float16
).to("cuda")
Text-to-image
For text-to-image, pass a text prompt. By default, SDXL generates a 1024x1024 image for the best results. You can try setting the height and width parameters to 768x768 or 512x512, but anything below 512x512 is not likely to work.

Copied
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]
image
generated image of an astronaut in a jungle
Image-to-image
For image-to-image, SDXL works especially well with image sizes between 768x768 and 1024x1024. Pass an initial image, and a text prompt to condition the image with:

Copied
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
generated image of a dog catching a frisbee in a jungle
Inpainting
For inpainting, you’ll need the original image and a mask of what you want to replace in the original image. Create a prompt to describe what you want to replace the masked area with.

Copied
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
generated image of a deep sea diver in a jungle
Refine image quality
SDXL includes a refiner model specialized in denoising low-noise stage images to generate higher-quality images from the base model. There are two ways to use the refiner:

use the base and refiner models together to produce a refined image
use the base model to produce an image, and subsequently use the refiner model to add more details to the image (this is how SDXL was originally trained)
Base + refiner model
When you use the base and refiner model together to generate an image, this is known as an ensemble of expert denoisers. The ensemble of expert denoisers approach requires fewer overall denoising steps versus passing the base model’s output to the refiner model, so it should be significantly faster to run. However, you won’t be able to inspect the base model’s output because it still contains a large amount of noise.

As an ensemble of expert denoisers, the base model serves as the expert during the high-noise diffusion stage and the refiner model serves as the expert during the low-noise diffusion stage. Load the base and refiner model:

Copied
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
To use this approach, you need to define the number of timesteps for each model to run through their respective stages. For the base model, this is controlled by the denoising_end parameter and for the refiner model, it is controlled by the denoising_start parameter.

The denoising_end and denoising_start parameters should be a float between 0 and 1. These parameters are represented as a proportion of discrete timesteps as defined by the scheduler. If you’re also using the strength parameter, it’ll be ignored because the number of denoising steps is determined by the discrete timesteps the model is trained on and the declared fractional cutoff.

Let’s set denoising_end=0.8 so the base model performs the first 80% of denoising the high-noise timesteps and set denoising_start=0.8 so the refiner model performs the last 20% of denoising the low-noise timesteps. The base model output should be in latent space instead of a PIL image.

Copied
prompt = "A majestic lion jumping from a big stone at night"

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
).images[0]
image
generated image of a lion on a rock at night
default base model
generated image of a lion on a rock at night in higher quality
ensemble of expert denoisers
The refiner model can also be used for inpainting in the StableDiffusionXLInpaintPipeline:

Copied
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import torch

base = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

image = base(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
).images[0]
make_image_grid([init_image, mask_image, image.resize((512, 512))], rows=1, cols=3)
This ensemble of expert denoisers method works well for all available schedulers!

Base to refiner model
SDXL gets a boost in image quality by using the refiner model to add additional high-quality details to the fully-denoised image from the base model, in an image-to-image setting.

Load the base and refiner models:

Copied
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
You can use SDXL refiner with a different base model. For example, you can use the Hunyuan-DiT or PixArt-Sigma pipelines to generate images with better prompt adherence. Once you have generated an image, you can pass it to the SDXL refiner model to enhance final generation quality.

Generate an image from the base model, and set the model output to latent space:

Copied
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = base(prompt=prompt, output_type="latent").images[0]
Pass the generated image to the refiner model:

Copied
image = refiner(prompt=prompt, image=image[None, :]).images[0]
generated image of an astronaut riding a green horse on Mars
base model
higher quality generated image of an astronaut riding a green horse on Mars
base model + refiner model
For inpainting, load the base and the refiner model in the StableDiffusionXLInpaintPipeline, remove the denoising_end and denoising_start parameters, and choose a smaller number of inference steps for the refiner.

Micro-conditioning
SDXL training involves several additional conditioning techniques, which are referred to as micro-conditioning. These include original image size, target image size, and cropping parameters. The micro-conditionings can be used at inference time to create high-quality, centered images.

You can use both micro-conditioning and negative micro-conditioning parameters thanks to classifier-free guidance. They are available in the StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, and StableDiffusionXLControlNetPipeline.

Size conditioning
There are two types of size conditioning:

original_size conditioning comes from upscaled images in the training batch (because it would be wasteful to discard the smaller images which make up almost 40% of the total training data). This way, SDXL learns that upscaling artifacts are not supposed to be present in high-resolution images. During inference, you can use original_size to indicate the original image resolution. Using the default value of (1024, 1024) produces higher-quality images that resemble the 1024x1024 images in the dataset. If you choose to use a lower resolution, such as (256, 256), the model still generates 1024x1024 images, but they’ll look like the low resolution images (simpler patterns, blurring) in the dataset.

target_size conditioning comes from finetuning SDXL to support different image aspect ratios. During inference, if you use the default value of (1024, 1024), you’ll get an image that resembles the composition of square images in the dataset. We recommend using the same value for target_size and original_size, but feel free to experiment with other options!

🤗 Diffusers also lets you specify negative conditions about an image’s size to steer generation away from certain image resolutions:

Copied
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_target_size=(1024, 1024),
).images[0]

Images negatively conditioned on image resolutions of (128, 128), (256, 256), and (512, 512).
Crop conditioning
Images generated by previous Stable Diffusion models may sometimes appear to be cropped. This is because images are actually cropped during training so that all the images in a batch have the same size. By conditioning on crop coordinates, SDXL learns that no cropping - coordinates (0, 0) - usually correlates with centered subjects and complete faces (this is the default value in 🤗 Diffusers). You can experiment with different coordinates if you want to generate off-centered compositions!

Copied
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt, crops_coords_top_left=(256, 0)).images[0]
image
generated image of an astronaut in a jungle, slightly cropped
You can also specify negative cropping coordinates to steer generation away from certain cropping parameters:

Copied
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_crops_coords_top_left=(0, 0),
    negative_target_size=(1024, 1024),
).images[0]
image
Use a different prompt for each text-encoder
SDXL uses two text-encoders, so it is possible to pass a different prompt to each text-encoder, which can improve quality. Pass your original prompt to prompt and the second prompt to prompt_2 (use negative_prompt and negative_prompt_2 if you’re using negative prompts):

Copied
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2).images[0]
image
generated image of an astronaut in a jungle in the style of a van gogh painting
The dual text-encoders also support textual inversion embeddings that need to be loaded separately as explained in the SDXL textual inversion section.

Optimizations
SDXL is a large model, and you may need to optimize memory to get it to run on your hardware. Here are some tips to save memory and speed up inference.

Offload the model to the CPU with enable_model_cpu_offload() for out-of-memory errors:
Copied
- base.to("cuda")
- refiner.to("cuda")
+ base.enable_model_cpu_offload()
+ refiner.enable_model_cpu_offload()
Use torch.compile for ~20% speed-up (you need torch>=2.0):
Copied
+ base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
+ refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
Enable xFormers to run SDXL if torch<2.0:
Copied
+ base.enable_xformers_memory_efficient_attention()
+ refiner.enable_xformers_memory_efficient_attention()
Other resources
If you’re interested in experimenting with a minimal version of the UNet2DConditionModel used in SDXL, take a look at the minSDXL implementation which is written in PyTorch and directly compatible with 🤗 Diffusers.

<
>
Update on GitHub
