#!/bin/bash

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="/home/work/main/sanghoon/my_folder/pika" \
  --output_dir="/home/work/main/sanghoon/my_folder/train_model" \
  --instance_prompt="a photo of pika" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1200 \
  —report_to="wandb" \
  —validation_prompt="a photo of pika, masterpiece, highly detailed" \

