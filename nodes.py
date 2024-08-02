import fal_client
import torch
import requests
import numpy as np
from PIL import Image
import io
import os

class FluxAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "endpoint": (["schnell (4+ steps)", "dev (25 steps)", "pro (25 steps)"],),
                "resolution": (["1024x1024 (1:1)", "512x512 (1:1)", "768x1024 (4:3)", "576x1024 (9:16)", "1024x720 (3:4)", "1024x576 (16:9)"],),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215})
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FalAPI"

    def generate_image(self, prompt, endpoint, resolution, steps, api_key, seed,):
        #set endpoint
        if endpoint == "schnell (4+ steps)":
            endpoint = "fal-ai/flux/schnell"
        elif endpoint == "pro (25 steps)":
            endpoint = "fal-ai/flux-pro"
        else:
            endpoint = "fal-ai/flux/dev"
        #convert dimensions
        AR = {
        "1024x1024 (1:1)": "square_hd",
        "512x512 (1:1)": "square",
        "768x1024 (4:3)": "portrait_4_3",
        "576x1024 (9:16)": "portrait_16_9",
        "1024x720 (3:4)": "landscape_4_3",
        "1024x576 (16:9)": "landscape_16_9",
        }
        image_size = AR.get(resolution)
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        handler = fal_client.submit(
        endpoint,
        arguments={
            "prompt": prompt,
            "seed": seed,
            "image_size": image_size,
            "num_inference_steps": steps,
            "num_images": 1,}  #Hardcoded to 1 for now
        )
        result = handler.get()
        image_url = result['images'][0]['url']
        #Download the image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        #make image more comfy
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "FluxAPI": FluxAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAPI": "FluxAPI",
}