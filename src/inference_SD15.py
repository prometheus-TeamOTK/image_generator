from diffusers import DiffusionPipeline
import torch
import os 

pipe_id = "stablediffusionapi/meinaalter"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("./models/lora/1.5", weight_name="Twenty Fith Bam.safetensors", adapter_name="TOG")
pipe.load_lora_weights("./models/lora/1.5", weight_name="wanostyle_2_offset.safetensors", adapter_name="onePiece")

pipe.safety_checker = None
pipe.requires_safety_checker = False

pipe.set_adapters(adapter_names=["TOG", "onePiece"],adapter_weights=[1,0])

# prompt = "cinematic photo casual Snow white, <lora:add-detail-xl:1> <lora:princess_xl_v2:0.9>, . 35mm photograph, film, bokeh, professional, 4k, highly detailed, in the forest, sunny, summer, dress,smile"
prompt = "twenty-fifth bam, 1boy, red vest, black shirt, long sleeve, scarf, solo, looking at viewer, upper body, potrait, tower, cityscape, sky, spacecraft, skyscraper, (masterpiece:1.2, best quality), smile"
negative_prompt = "(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy, watermark, signature, text, logo"
# negative_prompt=negative_prompt,
lora_scale= 1
image = pipe(prompt,  num_inference_steps=40, cross_attention_kwargs={"scale": lora_scale}).images[0]

image.save('./output/img15.png')