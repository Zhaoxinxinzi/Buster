import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from utils.encoder_utils import compute_text_embeddings
import torch
from datasets import load_dataset
import json
import sys

dataset = load_dataset("path_to/datasets/ChristophSchuhmann/improved_aesthetics_6.5plus")
dataset_clean = []
for i in range(5000):
    dataset_clean.append(dataset['train'][i]['TEXT'])

dirty_file = "path_to/datasets/i2p/i2p_sexual_prompts.txt"
with open(dirty_file) as f:
    dataset_dirty = f.read().splitlines()
    


sample_size_clean = 500
sample_size_dirty = 500
# data_clean = dataset_clean
data_dirty = dataset_dirty
data_clean = np.random.choice(dataset_clean, size=min(sample_size_clean, len(dataset_clean)), replace=False)
# data_dirty = np.random.choice(dataset_dirty, size=min(sample_size_dirty, len(dataset_dirty)), replace=False)
        
tokenizer = CLIPTokenizer.from_pretrained(
            "path_to/model/openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
            "path_to/model/openai/clip-vit-large-patch14")
torch_device = "cuda"
text_encoder.to(torch_device)


clean_embeddings = []
for sample in data_clean:
    text_clean = tokenizer(sample,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    with torch.no_grad():
        embedding_clean = text_encoder(text_clean.input_ids.cuda())[0]
    clean_embeddings.append(embedding_clean.cpu().detach().numpy())

dirty_embeddings = []
for sample in data_dirty:
    text_dirty = tokenizer(sample,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    with torch.no_grad():
        embedding_dirty = text_encoder(text_dirty.input_ids.cuda())[0]
    dirty_embeddings.append(embedding_dirty.cpu().detach().numpy())
    

all_embeddings = np.concatenate((clean_embeddings, dirty_embeddings), axis=0)
all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

labels = np.array(['clean'] * len(clean_embeddings) + ['dirty'] * len(dirty_embeddings))


tsne = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne.fit_transform(all_embeddings)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


clean_indices = np.where(labels == 'clean')[0]
dirty_indices = np.where(labels == 'dirty')[0]

ax.scatter(embeddings_3d[clean_indices, 0], embeddings_3d[clean_indices, 1], embeddings_3d[clean_indices, 2], c='b', label='Clean', s=50)
ax.scatter(embeddings_3d[dirty_indices, 0], embeddings_3d[dirty_indices, 1], embeddings_3d[dirty_indices, 2], c='r', label='Dirty', s=50)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend(loc='upper left')
plt.savefig("visualize/c-aesthetics500-d-i2psexual500.png")
