import os
import random
import zipfile
from typing import Any, Dict

import torch
from diffusers import DiffusionPipeline
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


class DiffuserPipeline:
    def __init__(self) -> None:
        self.adapter_map = {
            "snow_white": "princess",
            "elsa": "princess",
            "blossom": "powerpuff",
            "naruto": "naruto",
            "bam": "TOG",
            "luffy": "onePiece",
        }

        self.presetPrompt = {
            "snow_white": "Snow White",
            "elsa": "Elsa",
            "blossom": "ppg, blossom",
            "naruto": "Naruto",
            "bam": "twenty-fifth bam, 1boy, red vest, black shirt, long sleeve, scarf",
            "luffy": "wanostyle, monkey d luffy",
        }

        self.sdxl_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        self.sdxl_pipe.load_lora_weights(
            "../lora/XL",
            weight_name="princess_xl_v2.safetensors",
            adapter_name="princess",
        )
        self.sdxl_pipe.load_lora_weights(
            "../lora/XL",
            weight_name="ppg_v04.safetensors",
            adapter_name="powerpuff",
        )
        self.sdxl_pipe.load_lora_weights(
            "../lora/XL",
            weight_name="narutov3.safetensors",
            adapter_name="naruto",
        )

        self.sd1_5_pipe = DiffusionPipeline.from_pretrained(
            "Lykon/AnyLoRA", torch_dtype=torch.float16
        ).to("cuda")
        self.sd1_5_pipe.load_lora_weights(
            "../lora/1.5",
            weight_name="Twenty Fith Bam.safetensors",
            adapter_name="TOG",
        )
        self.sd1_5_pipe.load_lora_weights(
            "../lora/1.5",
            weight_name="wanostyle_2_offset.safetensors",
            adapter_name="onePiece",
        )
        self.sd1_5_pipe.safety_checker = None
        self.sd1_5_pipe.requires_safety_checker = False

    def __call__(self, data: Dict[str, Any]) -> Response:
        bot = data["bot"]
        summary = data["summary"]

        if isinstance(summary, str):
            summary = [summary]

        if bot in ["snow_white", "elsa", "blossom", "naruto"]:
            self.sdxl_pipe.set_adapters(self.adapter_map[bot])
            pipe = self.sdxl_pipe
        elif bot in ["bam", "luffy"]:
            self.sd1_5_pipe.set_adapters(self.adapter_map[bot])
            pipe = self.sd1_5_pipe
        else:
            return

        # Hyperparameters
        lora_scale = 1.0
        negative_prompt = "nsfw, bad hands, text, worst quality, cropped, blurry, ugly, extra arms, cross-eye"
        num_inference_steps = 30
        seed = random.randint(0, 9007199254740991)

        image_paths = []
        for i in range(len(summary)):
            image = pipe(
                f"{self.presetPrompt[bot]}, {summary[i]}",
                num_inference_steps=num_inference_steps,
                cross_attention_kwargs={"scale": lora_scale},
                generator=torch.Generator("cuda").manual_seed(seed),
                negative_prompt=negative_prompt,
            ).images[0]

            image_path = f"../output/image_{i}.png"
            image_paths.append(image_path)
            image.save(image_path)

        zip_path = f"../output/image_{seed}.zip"

        # Create a zip file
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for image in image_paths:
                zipf.write(image, os.path.basename(image))

        return zip_path


@app.route("/genimage", methods=["POST"])
def generate_image():
    response = Response()
    if request.method == "POST":
        response.headers.add("Access-Control-Allow-Origin", "*")
        data = request.get_json()

        zip_path = DiffuserPipeline(data)

        return send_file(zip_path, mimetype="application/zip", as_attachment=True)


@app.route("/")
def index():
    return "prometheus image generator API"


if __name__ == "__main__":
    # Run the Flask app
    DiffuserPipeline = DiffuserPipeline()
    app.run(host="0.0.0.0", port=5001)
