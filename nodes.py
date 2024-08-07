import fal_client
import replicate
import base64
import torch
import requests
import numpy as np
from PIL import Image
import io
import os

class FalAuraFlowAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1, "max": 20, "step": 0.5, "forceInput": False}),
                "expand_prompt": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, prompt, steps, api_key, seed, cfg, expand_prompt):
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        handler = fal_client.submit(
        "fal-ai/aura-flow",
        arguments={
            "prompt": prompt,
            "seed": seed,
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "num_images": 1, #Hardcoded to 1 for now
            "expand_prompt": expand_prompt,}
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

class FalFluxI2IAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True,}),
                "prompt": ("STRING", {"multiline": True}),
                "strength": ("FLOAT", {"default": 0.90, "min": 0.01, "max": 1, "step": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1, "max": 20, "step": 0.5, "forceInput": False}),
                "no_downscale": ("BOOLEAN", {"default": False,}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, image, prompt, strength, steps, api_key, seed, cfg, no_downscale):
        #Convert from image tensor to array
        image_np = 255. * image.cpu().numpy().squeeze()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_np).convert('L')
        #downscale image to prevent excess cost
        width, height = img.size #get size for checking
        max_dimension = max(width, height)
        scale_factor = 1024 / max_dimension
        if scale_factor < 1 and not no_downscale:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        width, height = img.size #get size for api
        #upload image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        file = buffered.getvalue()
        image_url = fal_client.upload(file, "image/png")
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        handler = fal_client.submit(
        "fal-ai/flux/dev/image-to-image",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "image_size": {
                "width": width,
                "height": height},
            "strength": strength,
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "num_images": 1, #Hardcoded to 1 for now
        })
        result = handler.get()
        image_url = result['images'][0]['url']
        #Download the image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        #make image more comfy
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        return (output_image,)

class FluxResolutionPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (["1024x1024 (1:1)", "512x512 (1:1)", "832x1216 (2:3)", "1216x832 (3:2)", "768x1024 (4:3)", "1024x720 (3:4)", "896x1088 (4:5)", "1088x896 (5:4)", "576x1024 (9:16)", "1024x576 (16:9)"],),
            },
        }
    
    RETURN_TYPES = ("INT","INT",)
    FUNCTION = "set_resolution"
    CATEGORY = "ComfyCloudAPIs"

    def set_resolution(self, aspect_ratio):
        ar = {
            "1024x1024 (1:1)": (1024, 1024),
            "512x512 (1:1)": (512, 512),
            "832x1216 (2:3)": (832, 1216),
            "1216x832 (3:2)": (1216, 832),
            "768x1024 (4:3)": (768, 1024),
            "1024x720 (3:4)": (1024, 720),
            "896x1088 (4:5)": (896, 1088),
            "1088x896 (5:4)": (1088, 896),
            "576x1024 (9:16)": (576, 1024),
            "1024x576 (16:9)": (1024, 576)
        }
        width, height = ar.get(aspect_ratio)
        return (width, height,)

class FalFluxAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "endpoint": (["schnell (4+ steps)", "dev (25 steps)", "pro (25 steps)"],),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg_dev_and_pro": ("FLOAT", {"default": 3.5, "min": 1, "max": 20, "step": 0.5, "forceInput": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, prompt, endpoint, width, height, steps, api_key, seed, cfg_dev_and_pro):
        #prevent too many steps error
        if endpoint == "schnell (4+ steps)" and steps > 8:
            steps = 8
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
    CATEGORY = "ComfyCloudAPIs"

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
    "FluxResolutionPresets": FluxResolutionPresets,
    "FalAuraFlowAPI": FalAuraFlowAPI,
    "FalFluxI2IAPI": FalFluxI2IAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalFluxAPI": "FalFluxAPI",
    "ReplicateFluxAPI": "ReplicateFluxAPI",
    "FluxResolutionPresets": "FluxResolutionPresets",
    "FalAuraFlowAPI": "FalAuraFlowAPI",
    "FalFluxI2IAPI": "FalFluxI2IAPI",
}