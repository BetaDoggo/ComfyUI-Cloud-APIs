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
        #set endpoint and inputs
        if model == "schnell":
            model = "black-forest-labs/flux-schnell"
            input={
            "prompt": prompt,
            "seed": seed,
            "output_format": "png",
            "disable_safety_checker": True,
            "aspect_ratio": aspect_ratio,}
        elif model == "pro":
            model = "black-forest-labs/flux-pro"
            if cfg_dev_and_pro > 5: #pro only supports cfg 1-5
                cfg_dev_and_pro = 5
            input={
            "prompt": prompt,
            "steps": steps_pro,
            "output_format": "png",
            "safety_tolerance": 5, #lowest value
            "aspect_ratio": aspect_ratio,
            "guidance": cfg_dev_and_pro,
            "interval": creativity_pro,}   
        else:
            model = "black-forest-labs/flux-dev"
            input={
            "prompt": prompt,
            "seed": seed,
            "output_format": "png",
            "disable_safety_checker": True,
            "aspect_ratio": aspect_ratio,
            "guidance": cfg_dev_and_pro,}
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["REPLICATE_API_TOKEN"] = key
        #make request
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