# Stable Diffusion Image Generator

This is a simple Python application that uses the Stable Diffusion model to generate images based on text prompts.

## Features

- Downloads and runs the Stable Diffusion model locally
- Allows users to provide text prompts for image generation
- Saves generated images to a local 'Generated' folder
- Supports both GPU and CPU execution

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Transformers
- Accelerate
- Pillow

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
python main.py
```

You will be prompted to enter a text description of the image you want to generate. After processing, the image will be saved to the 'Generated' folder.

## Notes

- The first run will download the Stable Diffusion model (about 4GB), which may take some time depending on your internet connection.
- GPU acceleration is used if available, otherwise the application will fall back to CPU (which will be much slower).
- Image generation can take several seconds to a few minutes depending on your hardware.
