# clip_utils.py

import torch
from transformers import CLIPTokenizer, CLIPModel
import torch.nn as nn


def load_clip_with_pad_token(model_name="openai/clip-vit-base-patch32", pad_token="[PAD]"):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    # Add <pad> token
    new_tokens = tokenizer.add_special_tokens({'pad_token': pad_token})

    if new_tokens > 0:
        old_emb = model.text_model.embeddings.token_embedding
        new_emb = nn.Embedding(len(tokenizer), old_emb.embedding_dim)
        new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
        model.text_model.embeddings.token_embedding = new_emb

    return tokenizer, model
