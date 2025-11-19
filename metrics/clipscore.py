from torchmetrics.functional.multimodal import clip_score
from functools import partial
import os
import torch
from PIL import Image
import clip
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_score_fn = partial(clip_score, model_name_or_path="path_to/model/openai/clip-vit-large-patch14")

def calculate_clip_score(images, prompts):
    images_int = (np.array(images) * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def read_prompts(file_path, num_texts):
    prompts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for i in range(num_texts):
            line = file.readline()
            line = line[:77]
            if not line:
                break
            prompts.append(line.strip())
    return prompts

def read_images(image_folder, num_images):
    images = []
    for i in range(num_images):
        # image_path = os.path.join(image_folder, f"img_{i:02d}.png")
        image_path = os.path.join(image_folder, f"sd14_output_{i}.png")
        image = Image.open(image_path)
        image = np.array(image)
        images.append(image)
    return images


num = 100
prompt_path = "path_to/prompts/captions_10000.txt"
image_path = "outputs/sd14-coco"
prompts = read_prompts(prompt_path,num)
images = read_images(image_path,num)

sd_clip_score = calculate_clip_score(images, prompts)
print(f"{image_path} CLIP score: {sd_clip_score}")

