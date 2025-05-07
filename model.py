# model.py

from transformers import CLIPModel, CLIPTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPProcessor


DROPOUT = 0.1


class CLIPBackbone(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        out = self.clip.vision_model(pixel_values).last_hidden_state
        return out[:, 1:, :]

    @torch.no_grad()
    def embed_text(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        emb = self.clip.text_model.embeddings
        return emb(input_ids=input_ids)


def build_attention_mask(image_len: int, text_len: int, device: torch.device) -> torch.Tensor:
    L = image_len + text_len
    mask = torch.ones(L, L, device=device)
    mask[image_len:, image_len:] = torch.tril(mask[image_len:, image_len:])
    return mask


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, max_seq_len: int = 77):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('causal_mask', torch.tril(
            torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, image_len: int, pad_mask: torch.Tensor, attn_mask: torch.Tensor):
        B, L, D = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attn_scores = q @ k.transpose(-2, -1) * D**-0.5

        attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_scores = attn_scores.masked_fill(
            pad_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embeded_dim: int, head_size: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 77):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(embeded_dim, head_size, max_seq_len) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, embeded_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, image_len: int, pad_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            head(x, image_len=image_len, pad_mask=pad_mask, attn_mask=attn_mask)
            for head in self.heads
        ], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embeded_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 77+49):
        super().__init__()
        head_size = embeded_dim // num_heads
        self.attn = MultiHeadAttention(
            embeded_dim=embeded_dim,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.ln1 = nn.LayerNorm(embeded_dim)
        self.ln2 = nn.LayerNorm(embeded_dim)
        self.mlp = MLP(embeded_dim)

    def forward(self, x: torch.Tensor, image_len: int, pad_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), image_len=image_len,
                          pad_mask=pad_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MMTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, img_dim: int = 768,
                 num_blocks: int = 4, num_heads: int = 8, max_seq_len: int = 77):
        super().__init__()
        self.backbone = CLIPBackbone()
        self.image_proj = nn.Linear(img_dim, embed_dim)
        self.txt_proj = nn.Identity()
        self.pos_embed = nn.Embedding(max_seq_len + 49, embed_dim)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads,
                               max_seq_len=max_seq_len + 49)
            for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, input_ids, attention_mask):
        B, L = input_ids.shape
        img_hidden = self.backbone.encode_image(images)
        txt_embed = self.txt_proj(self.backbone.embed_text(input_ids))

        img_proj = self.image_proj(img_hidden)

        img_pos = torch.arange(img_proj.size(
            1), device=images.device).unsqueeze(0)
        txt_pos = torch.arange(L, device=images.device).unsqueeze(0)
        img_proj = img_proj + self.pos_embed(img_pos)
        txt_embed = txt_embed + self.pos_embed(txt_pos + img_proj.size(1))

        x = torch.cat([img_proj, txt_embed], dim=1)

        img_mask = torch.ones(B, img_proj.size(
            1), dtype=attention_mask.dtype, device=images.device)
        pad_mask = torch.cat([img_mask, attention_mask], dim=1)

        attn_mask = build_attention_mask(img_proj.size(1), L, x.device)

        for block in self.blocks:
            x = block(x, image_len=img_proj.size(1),
                      pad_mask=pad_mask, attn_mask=attn_mask)

        x = self.ln_final(x)
        logits = self.output_layer(x[:, img_proj.size(1):])
        return logits
