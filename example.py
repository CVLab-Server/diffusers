import argparse
import os
from diffusers import StableDiffusionXLPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description='SDXL Text-to-Image Generator')
    parser.add_argument('--captions', type=str, default='/usr/src/captions.txt', help='Path to caption text file')
    parser.add_argument('--output', type=str, default='/usr/src/output', help='Output directory path')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the SDXL pipeline
    print("Loading SDXL model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True,
        add_watermarker=False  # Disable watermarker to avoid opencv issues
    ).to("cuda")
    
    # Read captions from file
    with open(args.captions, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f if line.strip()]
    
    # Generate images for each caption
    for i, caption in enumerate(captions):
        print(f"Generating image {i+1}/{len(captions)}: {caption}")
        
        # Generate image
        image = pipeline(prompt=caption).images[0]
        
        # Create filename from caption (replace spaces with underscores)
        filename = caption.replace(' ', '_')[:100] + '.png'  # Limit filename length
        filepath = os.path.join(args.output, filename)
        
        # Save image
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"All {len(captions)} images generated successfully!")

if __name__ == "__main__":
    main()