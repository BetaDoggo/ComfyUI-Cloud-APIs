import fal_client
import replicate
import torch
import requests
import numpy as np
from PIL import Image
import io
import os

class FalFluxAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "endpoint": (["schnell (4+ steps)", "dev (25 steps)", "pro (25 steps)"],),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64, "forceInput": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64, "forceInput": False}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg_dev_and_pro": ("FLOAT", {"default": 3.5, "min": 1, "max": 20, "step": 0.5, "forceInput": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FalAPI"

    def generate_image(self, prompt, endpoint, width, height, steps, api_key, seed, cfg_dev_and_pro):
        #set endpoint
        models = {
            "schnell (4+ steps)": "fal-ai/flux/schnell",
            "pro (25 steps)": "fal-ai/flux-pro",
            }
        endpoint = models.get(endpoint, "fal-ai/flux/dev")
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
            "guidance_scale": cfg_dev_and_pro,
            "safety_tolerance": 6,
            "image_size": {
                "width": width,
                "height": height,
            },
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

class ReplicateFluxAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["schnell", "dev", "pro"],),
                "aspect_ratio": (["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg_dev_and_pro": ("FLOAT", {"default": 3.5, "min": 1, "max": 10, "step": 0.5, "forceInput": False}),
                "steps_pro": ("INT", {"default": 25, "min": 1, "max": 50}),
                "creativity_pro": ("INT", {"default": 2, "min": 1, "max": 4}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ReplicateAPI"

    def generate_image(self, prompt, model, aspect_ratio, api_key, seed, cfg_dev_and_pro, steps_pro, creativity_pro,):
        #set endpoint
        models = {
            "schnell": "black-forest-labs/flux-schnell",
            "pro": "black-forest-labs/flux-pro",
        }
        model = models.get(model, "black-forest-labs/flux-dev")
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["REPLICATE_API_TOKEN"] = key
        #make request
        input={
            "prompt": prompt,
            "steps": steps_pro,
            "seed": seed,
            "disable_safety_checker": True,
            "output_format": "png",
            "safety_tolerance": 5, #lowest value
            "aspect_ratio": aspect_ratio,
            "guidance": cfg_dev_and_pro,
            "interval": creativity_pro,}  
        output = replicate.run(model, input=input)
        image_url = output
        #Download the image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        #make image more comfy
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "FalFluxAPI": FalFluxAPI,
    "ReplicateFluxAPI": ReplicateFluxAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalFluxAPI": "FalFluxAPI",
    "ReplicateFluxAPI": "ReplicateFluxAPI",
}