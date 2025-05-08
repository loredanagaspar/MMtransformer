# file: transformer_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class SelfAttentionDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=49409, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 77, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tgt_ids, tgt_key_padding_mask=None):
        # memory: [N, 126, 512] (image+caption embedding)
        # tgt_ids: [N, T] (caption input_ids)
        # tgt_key_padding_mask: [N, T] -> True for PADs

        tgt_emb = self.token_embedding(tgt_ids) + self.pos_embedding  # [N, T, D]
        tgt_emb = tgt_emb.transpose(0, 1)  # [T, N, D]
        memory = memory.transpose(0, 1)    # [S, N, D]

        T = tgt_ids.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_ids.device)  # [T, T]

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.out_proj(out.transpose(0, 1))  # [N, T, vocab_size]
        return logits


def load_batch(batch_size=16):
    combined = torch.tensor(np.load("combined_embeddings.npy"))  # [N, 126, 512]
    with open("captions_tokenized.json") as f:
        cap_data = json.load(f)
        input_ids = [d["tokens"] for d in cap_data]
        input_ids = torch.tensor(input_ids)  # [N, 77]

    # Padding mask: 1 where PAD
    pad_id = 49408
    tgt_pad_mask = (input_ids == pad_id)  # [N, 77]

    return combined[:batch_size], input_ids[:batch_size], tgt_pad_mask[:batch_size]


if __name__ == "__main__":
    model = SelfAttentionDecoder()
    combined, input_ids, pad_mask = load_batch()

    logits = model(combined, input_ids, pad_mask)  # [B, T, V]
    print("✅ logits:", logits.shape)  # [B, T, vocab_size]
