# ComfyUI-Cloud-APIs
I wanted to use some larger models in my local ComfyUI workflows but I'm too gpu poor. Currently model support is limited but I'll add more when I find the time.
# Supported Models:
- Flux (fal and replicate)
- Auraflow (fal)
# Quick Setup
1. Install extension and dependencies (preferably with comfy manager)
2. Place your api key/token in a text file in ComfyUI-Cloud-APIs/keys
# Step by step setup (Fal)
1. Install the extension with the "Install via Git URL" option in comfyUI manager
2. Create an account at [fal.ai](https://fal.ai/)
3. Go to https://fal.ai/dashboard/keys and click "Add key"
4. Name the key then copy it into a text file in the ComfyUI-Cloud-APIs/keys folder. (there is a placeholder nokey.txt file which you can delete)
5. Consult the [models](https://fal.ai/models) page to get an idea of how much each generation will cost
6. Go to https://fal.ai/dashboard/billing and top up your account. For your financial well-being I recommend against automated topups, but I can't stop you.
# Step by step setup (Replicate)
1. Install the extension with the "Install via Git URL" option in comfyUI manager
2. Create an account at https://replicate.com/
3. Go to https://replicate.com/account/api-tokens and copy your token (or create a new one)
4. Copy the token into a text file in the ComfyUI-Cloud-APIs/keys folder. (there is a placeholder nokey.txt file which you can delete)
5. Consult https://replicate.com/explore to get an idea of how much each generation will cost
6. Go to https://replicate.com/account/billing to setup billing when you run out of free usage.
![preview](https://github.com/BetaDoggo/ComfyUI-fal-api/blob/main/preview.png)
