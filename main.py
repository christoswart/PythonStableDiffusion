import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time
import gradio as gr

# Check if the Generated folder exists, if not create it
if not os.path.exists('Generated'):
    os.makedirs('Generated')

def load_model():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = "cuda"
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU for processing (this will be slow).")
    
    # Load the Stable Diffusion model
    print("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        # Load the model with appropriate settings
        if cuda_available:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                # Removed the problematic revision parameter
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
            
        # Move model to device
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if using CUDA
        if cuda_available:
            pipe.enable_attention_slicing()
            # Optional: Enable memory efficient attention
            try:
                # This is optional and requires xformers package
                # pipe.enable_xformers_memory_efficient_attention()
                pass
            except Exception as e:
                print(f"Note: Could not enable xformers optimization: {e}")
        
        print(f"Model loaded successfully on {device}")
        return pipe
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Function to generate an image from a prompt
def generate_image(pipe, prompt, filename=None):
    print(f"Generating image for prompt: '{prompt}'")
    start_time = time.time()
    
    # Generate the image
    # Adjust guidance_scale as needed (higher = more adherence to prompt, but less variation)
    image = pipe(
        prompt, 
        guidance_scale=7.5,
        num_inference_steps=50  # Increase for better quality, decrease for speed
    ).images[0]
    
    # Create a filename if none provided
    if filename is None:
        # Replace spaces and special characters for a valid filename
        safe_prompt = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in prompt)
        filename = f"{safe_prompt[:50].replace(' ', '_')}.png"
    
    # Save the image
    image_path = os.path.join('Generated', filename)
    image.save(image_path)
    
    elapsed_time = time.time() - start_time
    print(f"Image generated in {elapsed_time:.2f} seconds")
    print(f"Image saved to: {image_path}")
    return image_path

if __name__ == '__main__':
    try:
        # Load the model
        pipe = load_model()
        
        # Get user input
        #prompt = input("Enter your image generation prompt: ")
        
        # Generate the image
        #image_path = generate_image(pipe, prompt)
        gr.Interface.from_pipeline(pipe).launch()
        
        #print(f"Success! Image saved to: {image_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
