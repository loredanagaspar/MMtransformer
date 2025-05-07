# model.py

from transformers import CLIPModel, CLIPTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPProcessor
import traceback


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
        # Add explicit checks
        try:
            out = self.clip.vision_model(pixel_values).last_hidden_state
            # Return only the patch tokens (skip the CLS token)
            return out[:, 1:, :]
        except Exception as e:
            print(f"ERROR in encode_image: {e}")
            print(f"Input shape: {pixel_values.shape}")
            raise

    @torch.no_grad()
    def embed_text(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        try:
            emb = self.clip.text_model.embeddings
            return emb(input_ids=input_ids)
        except Exception as e:
            print(f"ERROR in embed_text: {e}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Max token ID: {input_ids.max().item()}")
            raise


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
        
        # CLIP has max sequence length of 77 and image patches of 49 (7x7)
        max_positions = max_seq_len + 49  # 77 + 49 = 126
        self.pos_embed = nn.Embedding(max_positions, embed_dim)
        print(f"Position embedding table size: {max_positions}")
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads,
                               max_seq_len=max_seq_len + 49)
            for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Extra debugging
        self.debug_mode = True

    def forward(self, images, input_ids, attention_mask):
        try:
            # IMPORTANT: Add debugging information
            print(f"DEBUG - Input shapes: images={images.shape}, input_ids={input_ids.shape}, mask={attention_mask.shape}")
            print(f"DEBUG - Images memory: {images.device}, contiguous: {images.is_contiguous()}")
            print(f"DEBUG - Image values range: min={images.min().item():.2f}, max={images.max().item():.2f}")
            
            # Check for NaNs
            if torch.isnan(images).any():
                print("WARNING: NaN values detected in image input")
            
            # Set a fixed size for image tokens - CLIP ViT-B/32 should have 7x7=49 patches
            expected_image_tokens = 49
            
            # Process image through CLIP vision encoder with detailed error handling
            try:
                B, C, H, W = images.shape
                img_hidden = self.backbone.encode_image(images)
                print(f"DEBUG - img_hidden shape: {img_hidden.shape}")
                
                # Check that the output matches expected size
                if img_hidden.shape[1] != expected_image_tokens:
                    print(f"WARNING: Expected {expected_image_tokens} image tokens, got {img_hidden.shape[1]}")
            except Exception as e:
                print(f"ERROR in backbone.encode_image: {e}")
                # Fallback to random features for debugging
                print("FALLBACK: Using random features for image")
                img_hidden = torch.randn(images.shape[0], expected_image_tokens, 768, device=images.device)
            
            # Process text through CLIP text encoder
            B, L = input_ids.shape
            txt_embed = self.txt_proj(self.backbone.embed_text(input_ids))
            print(f"[SHAPES] txt_embed: {txt_embed.shape}")
            
            # Project image features to text embedding space
            img_proj = self.image_proj(img_hidden)
            print(f"[SHAPES] img_proj: {img_proj.shape}")
            
            img_len = img_proj.size(1)
            
            # Position embedding section
            try:
                # Create fixed position embeddings directly
                img_pos_embed = torch.zeros_like(img_proj)
                txt_pos_embed = torch.zeros_like(txt_embed)
                
                # For each position, fill with the appropriate embedding
                for pos in range(img_len):
                    # Make sure the embedding index is a scalar tensor on GPU
                    pos_idx = torch.tensor([pos], device=images.device)
                    # Get embedding for this position
                    pos_embed = self.pos_embed(pos_idx)
                    # Apply to all batches at this position
                    img_pos_embed[:, pos, :] = pos_embed
                
                for pos in range(L):
                    # Offset for text positions
                    pos_idx = torch.tensor([pos + img_len], device=images.device)
                    pos_embed = self.pos_embed(pos_idx)
                    txt_pos_embed[:, pos, :] = pos_embed
                
                # Add position embeddings
                img_proj = img_proj + img_pos_embed
                txt_embed = txt_embed + txt_pos_embed
                
            except Exception as e:
                print(f"Error in position embedding: {e}")
                print("WARNING: Skipping position embeddings due to error")
                # Just continue without position embeddings
            
            # Concatenate image and text features
            x = torch.cat([img_proj, txt_embed], dim=1)
            print(f"[SHAPES] concatenated x: {x.shape}")
            
            # Create attention masks
            img_mask = torch.ones(B, img_len, dtype=attention_mask.dtype, device=images.device)
            pad_mask = torch.cat([img_mask, attention_mask], dim=1)
            
            # Build causal attention mask
            attn_mask = build_attention_mask(img_len, L, x.device)
            print(f"Image length: {img_len}, Text length: {L}")
            print(f"Attention mask shape: {attn_mask.shape}")
            print(f"Pad mask shape: {pad_mask.shape}")
            
            # Visualize a small portion of the mask (first few rows/cols)
            if attn_mask.shape[0] < 20:  # Only for small masks
                print("Attention mask sample:")
                print(attn_mask[:10, :10])  # Show top-left corner
            
            # Process through transformer blocks
            for i, block in enumerate(self.blocks):
                x = block(x, image_len=img_len,
                          pad_mask=pad_mask, attn_mask=attn_mask)
                if i == 0 and self.debug_mode:
                    print(f"DEBUG - After block 0: x shape={x.shape}, x min={x.min().item():.2f}, x max={x.max().item():.2f}")
            
            # Final processing
            x = self.ln_final(x)
            logits = self.output_layer(x[:, img_len:])
            print(f"[SHAPES] logits: {logits.shape}")
            
            return logits
            
        except Exception as e:
            print(f"ERROR in MMTransformer.forward: {e}")
            traceback.print_exc()
            raise