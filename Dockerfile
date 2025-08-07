FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install diffusers and dependencies with compatibility fixes
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface-hub

WORKDIR /usr/src

# Copy the entire project to /usr/src
COPY . /usr/src/

# Download the SDXL model during build
RUN python3 -c "import torch; from diffusers import StableDiffusionXLPipeline; print('Downloading SDXL model...'); pipeline = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16', use_safetensors=True); print('Model downloaded successfully!')"

CMD ["/bin/bash"]