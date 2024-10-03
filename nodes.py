import fal_client
import replicate
import json
import torch
import requests
import numpy as np
from PIL import Image
import io
import os
import websocket
import uuid

class FalLLaVAAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True,}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image"}),
                "max_tokens": ("INT", {"default": 64, "min": 16, "max": 512, "step": 1}),
                "temp": ("FLOAT", {"default": 0.2, "min": 0, "max": 1}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
                "model": (["LLavaV15_13B", "LLavaV16_34B"],),
                "api_key": (api_keys,),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "ComfyCloudAPIs"

    def describe_image(self, image, prompt, max_tokens, temp, top_p, model, api_key,):
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        models = {"LLavaV15_13B": "fal-ai/llavav15-13b",
                  "LLavaV16_34B": "fal-ai/llava-next"}
        endpoint = models.get(model)
        #Convert from image tensor to array
        image_np = 255. * image.cpu().numpy().squeeze()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        #upload image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        file = buffered.getvalue()
        image_url = fal_client.upload(file, "image/png")
        handler = fal_client.submit(
        endpoint,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
        })
        result = handler.get()
        output_text = result['output']
        return (output_text,)

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
                "cfg": ("FLOAT", {"default": 3.5, "min": 0, "max": 20, "step": 0.5, "forceInput": False}),
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

class FalStableCascadeAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, deformed",}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "first_stage_steps": ("INT", {"default": 20, "min": 1, "max": 50}),
                "second_stage_steps": ("INT", {"default": 10, "min": 1, "max": 24}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "decoder_guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 16777215,}),
            },
        }
   
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, prompt, negative_prompt, width, height, first_stage_steps, second_stage_steps, guidance_scale, decoder_guidance_scale, api_key, seed):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key

        handler = fal_client.submit(
            "fal-ai/stable-cascade",
            arguments={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": {
                    "width": width,
                    "height": height,
                },
                "first_stage_steps": first_stage_steps,
                "second_stage_steps": second_stage_steps,
                "guidance_scale": guidance_scale,
                "second_stage_guidance_scale": decoder_guidance_scale,
                "enable_safety_checker": False,
                "num_images": 1,
                "seed": seed,
            }
        )

        result = handler.get()
        image_url = result['images'][0]['url']
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        
        return (output_image,)

class FalSoteDiffusionAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "newest, extremely aesthetic, best quality,",}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "very displeasing, worst quality, monochrome, realistic, oldest",}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "first_stage_steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "second_stage_steps": ("INT", {"default": 10, "min": 1, "max": 24}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "decoder_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 16777215,}),
            },
        }
   
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, prompt, negative_prompt, width, height, first_stage_steps, second_stage_steps, guidance_scale, decoder_guidance_scale, api_key, seed):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["ANIME_STYLE_API_KEY"] = key

        handler = fal_client.submit(
            "fal-ai/stable-cascade/sote-diffusion",
            arguments={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": {
                    "width": width,
                    "height": height,
                },
                "first_stage_steps": first_stage_steps,
                "second_stage_steps": second_stage_steps,
                "guidance_scale": guidance_scale,
                "second_stage_guidance_scale": decoder_guidance_scale,
                "enable_safety_checker": False,
                "num_images": 1,
                "seed": seed,
            }
        )

        result = handler.get()
        image_url = result['images'][0]['url']
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        return (output_image,)

class FalAddLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_url": ("STRING", {"multiline": False}),
                "scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 4}),
            },
            "optional":{
                "loras": ("STRING", {"forceInput": True,}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_lora"
    CATEGORY = "ComfyCloudAPIs"

    def string_lora(self, lora_url, scale, loras=None):
        if loras is not None:
            lora_dict = json.loads(loras)
        else:
            lora_dict = {"loras": []}
        lora_dict["loras"].append({"path": lora_url, "scale": scale})
        output_loras = json.dumps(lora_dict)
        return (output_loras,)

class FalFluxLoraAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "loras": ("STRING", {"forceInput": True,}),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1, "max": 20, "step": 0.5, "forceInput": False}),
                "no_downscale": ("BOOLEAN", {"default": False,}),
                "i2i_strength": ("FLOAT", {"default": 0.90, "min": 0.01, "max": 1, "step": 0.01}),
            },
            "optional":{
                "image": ("IMAGE", {"forceInput": True,}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, loras, prompt, width, height, steps, api_key, seed, cfg, no_downscale, i2i_strength, image=None,):
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        full_args = {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "image_size": {
                "width": width,
                "height": height},
            "guidance_scale": cfg,
            "enable_safety_checker": False,
            "num_inference_steps": steps,
            "num_images": 1, #Hardcoded to 1 for now
        }
        loras = json.loads(loras)
        endpoint = "fal-ai/flux-lora"
        if image is not None:
            endpoint = "fal-ai/flux-lora/image-to-image"
            #Convert from image tensor to array
            image_np = 255. * image.cpu().numpy().squeeze()
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_np)
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
            #setup img2img
            i2i_args = {
                "image_url": image_url,
                "strength": i2i_strength,
            }
            full_args.update(i2i_args)
        full_args.update(loras)
        handler = fal_client.submit(endpoint, arguments= full_args)
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
        #Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        os.environ["FAL_KEY"] = key
        #Convert from image tensor to array
        image_np = 255. * image.cpu().numpy().squeeze()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_np)
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
            "enable_safety_checker": False,
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

class RunwareAddLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_air": ("STRING", {"multiline": False}),
                "weight": ("FLOAT", {"default": 1, "min": 0.1, "max": 4}),
            },
            "optional":{
                "loras": ("STRING", {"forceInput": True,}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_lora"
    CATEGORY = "ComfyCloudAPIs"

    def string_lora(self, lora_air, weight, loras=None):
        if loras is not None:
            lora_dict = json.loads(loras)
        else:
            lora_dict = {"lora": []}
        lora_dict["lora"].append({"model": lora_air, "weight": weight})
        output_loras = json.dumps(lora_dict)
        return (output_loras,)

class RunWareAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 16, "forceInput": False}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 16, "forceInput": False}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg": ("FLOAT", {"default": 7, "min": 0, "max": 30, "step": 0.5, "forceInput": False}),
                "model_air": ("STRING",), # this expects a model name formatted with civit's air system. They have their selection here: https://docs.runware.ai/en/image-inference/models#model-explorer
            },
            "optional": {
                "loras": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ComfyCloudAPIs"

    def generate_image(self, positive_prompt, negative_prompt, width, height, steps, api_key, seed, cfg, model_air, loras=None):
        # Set api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(current_dir, "keys"), api_key), 'r', encoding='utf-8') as file:
            key = file.read()
        # connect to api websocket
        ws = websocket.create_connection("wss://ws-api.runware.ai/v1")
        # authenticate
        auth_request = [
            {
                "taskType": "authentication",
                "apiKey": key,
            }
        ]
        ws.send(json.dumps(auth_request))
        auth_response = ws.recv()
        print("auth:" + auth_response)
        # create request
        image_request = [
            {
                "taskType": "imageInference",
                "taskUUID": str(uuid.uuid4()), # create a random uuidv4
                "outputType": "URL",
                "outputFormat": "PNG",
                "positivePrompt": positive_prompt,
                "negativePrompt": negative_prompt,  
                "height": height,
                "width": width,
                "model": model_air,
                "steps": steps,
                "seed": seed,
                "CFGScale": cfg,
                "numberResults": 1
            }
        ]
        
        if loras is not None:
            loras = json.loads(loras)
            image_request[0].update(loras)

        ws.send(json.dumps(image_request))
        response = ws.recv()
        print("runware response:" + response)
        result = json.loads(response)
        image_url = result['data'][0]['imageURL']
        # Download the image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        # Convert image to ComfyUI format
        image = np.array(image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(image)[None,]
        ws.close()
        return (output_image,)

class FalFluxAPI:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys = [f for f in os.listdir(os.path.join(current_dir, "keys")) if f.endswith('.txt')]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "endpoint": (["schnell (4+ steps)", "dev (25+ steps)", "pro 1.1", "realism (25+ steps)", "pro (25+ steps)",],),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16, "forceInput": False}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "api_key": (api_keys,),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "cfg_dev_and_pro": ("FLOAT", {"default": 3.5, "min": 0, "max": 20, "step": 0.5, "forceInput": False}),
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
            "pro (25+ steps)": "fal-ai/flux-pro",
            "realism (25+ steps)": "fal-ai/flux-realism",
            "pro 1.1": "fal-ai/flux-pro/v1.1",
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
            "safety_tolerance": 5,
            "image_size": {
                "width": width,
                "height": height,
            },
            "num_inference_steps": steps,
            "enable_safety_checker": False,
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
                "model": (["schnell", "dev", "pro 1.1", "pro"],),
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
        image_url = output[0] if isinstance(output, list) else output #replicate started returning a different format, this works for both
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
    "FalSoteDiffusionAPI": FalSoteDiffusionAPI,
    "FalStableCascadeAPI": FalStableCascadeAPI,
    "FalLLaVAAPI": FalLLaVAAPI,
    "FalFluxLoraAPI": FalFluxLoraAPI,
    "FalAddLora": FalAddLora,
    "RunWareAPI": RunWareAPI,
    "RunwareAddLora": RunwareAddLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalFluxAPI": "FalFluxAPI",
    "ReplicateFluxAPI": "ReplicateFluxAPI",
    "FluxResolutionPresets": "FluxResolutionPresets",
    "FalAuraFlowAPI": "FalAuraFlowAPI",
    "FalFluxI2IAPI": "FalFluxI2IAPI",
    "FalSoteDiffusionAPI": "FalSoteDiffusionAPI",
    "FalStableCascadeAPI": "FalStableCascadeAPI",
    "FalLLaVAAPI": "FalLLaVAAPI",
    "FalAddLora": "FalAddLora",
    "RunWareAPI": "RunWareAPI",
    "RunwareAddLora": "RunwareAddLora",
}