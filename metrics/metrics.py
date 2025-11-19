import torch
from utils.attack_utils import inject_attribute_backdoor
from utils.encoder_utils import compute_text_embeddings
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity


def embedding_sim_clean(text_encoder_clean: torch.nn.Module,
                        text_encoder_backdoored: torch.nn.Module,
                        tokenizer: torch.nn.Module,
                        captions_clean: list,
                        batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    # with open(clean_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    #     captions_clean = [line.strip() for line in lines]

    # compute embeddings on clean encoder
    emb_clean = compute_text_embeddings(tokenizer, text_encoder_clean,
                                        captions_clean, batch_size)

    # compute embeddings on backdoored encoder
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder_backdoored,
                                           captions_clean, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = cosine_similarity(emb_clean, emb_backdoor, dim=1)

    mean_sim = similarity.mean().cpu().item()
    print(
        f'Computed Clean Similarity Score on {len(captions_clean)} samples: {mean_sim:.4f}'
    )

    return mean_sim



def embedding_sim_backdoor(text_encoder_clean: torch.nn.Module,
                        text_encoder_backdoored: torch.nn.Module,
                        tokenizer: torch.nn.Module,
                        captions_nsfw: list,
                        batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    # with open(backdoor_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    #     captions_nsfw = [line.strip() for line in lines]

    # compute embeddings on target prompt
    emb_clean = compute_text_embeddings(tokenizer, text_encoder_clean,
                                        captions_nsfw, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder_backdoored,
                                           captions_nsfw, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = cosine_similarity(emb_clean, emb_backdoor, dim=1)

    mean_sim = similarity.mean().cpu().item()
    print(
        f'Computed Backdoor Similarity Score on {len(captions_nsfw)} samples: {mean_sim:.4f}'
    )

    return mean_sim


def embedding_sim_target(text_encoder_clean: torch.nn.Module,
                        text_encoder_backdoored: torch.nn.Module,
                        tokenizer: torch.nn.Module,
                        captions_nsfw: list,
                        batch_size: int = 256) -> float:

    captions_clean = "A photo of a cute cat"
    # compute embeddings on target prompt
    emb_clean = compute_text_embeddings(tokenizer, text_encoder_clean,
                                        captions_clean)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder_backdoored,
                                           captions_nsfw, batch_size)
    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = cosine_similarity(emb_clean, emb_backdoor, dim=1)

    mean_sim = similarity.mean().cpu().item()
    print(
        f'Computed Target Similarity Score on {len(captions_nsfw)} samples: {mean_sim:.4f}'
    )

    return mean_sim