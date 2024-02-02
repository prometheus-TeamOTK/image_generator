from diffusers import DiffusionPipeline
import torch
import random

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("./models/lora/XL", weight_name="princess_xl_v2.safetensors", adapter_name="princess")
pipe.load_lora_weights("./models/lora/XL", weight_name="ppg_v04.safetensors", adapter_name="powerpuff")
pipe.load_lora_weights("./models/lora/XL", weight_name="narutov3.safetensors", adapter_name="naruto")

pipe.set_adapters(["princess", "powerpuff","naruto"],[1, 0, 1])

# seed = random.randint(0, 9007199254740991)
seed = 5078131701373955
gen = torch.Generator("cuda").manual_seed(seed)

prompt = "Elsa talking to Naruto"
# negative_prompt = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
# negative_prompt=negative_prompt,
lora_scale= 1.0
image = pipe(
    prompt,  num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator = gen,
).images[0]

print("seed:",seed)
image.save('./output/imgXL2.png')