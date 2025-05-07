import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
import wandb
import os
from transformers import CLIPTokenizer, CLIPProcessor
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from model import MMTransformer
from dataset import get_flickr30k_splits, collate_fn

# Config - Define these before model initialization
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LR_DECAY = 0.9
EMBED_DIM = 512
MAX_SEQ_LEN = 77

# ===== Initialize tokenizer and processor =====
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Add PAD token safely
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
print("✅ Tokenizer vocab size:", len(tokenizer))
PAD_ID = tokenizer.pad_token_id
print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)

# Update vocab size after adding special tokens
VOCAB_SIZE = len(tokenizer)

def decode_caption(input_ids):
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

# ===== Initialize WandB =====
wandb.init(project="clip-captioning")

# ===== Load dataset =====
train_ds, val_ds = get_flickr30k_splits()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                      shuffle=False, collate_fn=collate_fn, num_workers=4)

# ===== Initialize model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Clear CUDA cache
print("Clearing CUDA cache...")
torch.cuda.empty_cache()
print("GPU memory after clearing cache:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
def print_gpu_memory_stats():
    print("\nGPU Memory Stats:")
    print(f"  Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")

# Call it at key points in your training loop
print_gpu_memory_stats()

model = MMTransformer(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)

# Check for model mismatch with tokenizer
if model.output_layer.out_features != VOCAB_SIZE:
    print(f"⚠️ MISMATCH: Tokenizer vocab ({VOCAB_SIZE}) != Output layer ({model.output_layer.out_features})")
    model.output_layer = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=True)
    print(f"✅ Fixed: Output layer resized to {model.output_layer.out_features}")

model = model.to(device)

# ===== Optimizer and scheduler =====
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

# ===== Debug function for memory monitoring =====
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# ===== Training Loop =====
for epoch in range(NUM_EPOCHS):
    # Print a sample batch for debugging
    batch_images, batch_input_ids, batch_attn_mask = next(iter(train_loader))
    print(f"Batch shapes - Images: {batch_images.shape}, Input IDs: {batch_input_ids.shape}, Attn Mask: {batch_attn_mask.shape}")
    print(f"Input ID range: min={batch_input_ids.min().item()}, max={batch_input_ids.max().item()}, vocab_size={VOCAB_SIZE}")
    print(f"Sample caption: {tokenizer.decode(batch_input_ids[0])}")
    
    # Enable gradient anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Training phase
    model.train()
    total_loss = 0
    for batch_idx, (images, input_ids, attn_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")):
        # Verify input IDs are within vocab range
        assert input_ids.max().item() < VOCAB_SIZE, f"Token ID out of bounds: {input_ids.max().item()} >= {VOCAB_SIZE}"
        
        # Print sample input for first batch
        if batch_idx == 0:
            print("🔍 Sample input_ids:", input_ids[0].tolist())
            print("🔍 Decoded:", tokenizer.decode(input_ids[0]))
            print_gpu_memory()
            
        # Move data to device
        images, input_ids, attn_mask = images.to(device), input_ids.to(device), attn_mask.to(device)
        
        # Forward pass
        logits = model(images, input_ids, attn_mask)
        
        # Calculate loss
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), input_ids.view(B*T), ignore_index=PAD_ID)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Monitor gradients periodically
        if batch_idx % 10 == 0:  # Every 10 batches
            total_norm = 0
            param_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    param_norms.append((p.shape, param_norm))
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm:.4f}")
            
            # Alert on potential issues
            if total_norm > 10.0:
                print("⚠️ Large gradient norm detected - potential explosion")
                # Log the layers with largest gradients
                sorted_norms = sorted(param_norms, key=lambda x: x[1], reverse=True)
                for i, (shape, norm) in enumerate(sorted_norms[:3]):
                    print(f"  Top {i+1}: shape {shape}, norm {norm:.4f}")
            
            if total_norm < 0.01:
                print("⚠️ Very small gradient norm - potential vanishing gradient")
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        # Examine output on first batch
        if batch_idx == 0:
            with torch.no_grad():
                # Get the predicted distributions for a few tokens
                token_idx = min(5, T-1)  # Look at the 5th token prediction (or last if shorter)
                token_probs = F.softmax(logits[0, token_idx], dim=0)
                top_k = torch.topk(token_probs, 5)
                print(f"Top 5 predictions for token {token_idx}:")
                for i, (idx, prob) in enumerate(zip(top_k.indices, top_k.values)):
                    token = tokenizer.convert_ids_to_tokens(idx.item())
                    print(f"  {i+1}. '{token}' (ID: {idx.item()}) - {prob.item():.4f}")
                
                # Check if the ground truth is in top predictions
                true_token_id = input_ids[0, token_idx].item()
                true_token = tokenizer.convert_ids_to_tokens(true_token_id)
                true_prob = token_probs[true_token_id].item()
                print(f"  Ground truth: '{true_token}' (ID: {true_token_id}) - {true_prob:.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss})
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
    
    # Evaluation phase
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for images, input_ids, attn_mask in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            images, input_ids, attn_mask = images.to(device), input_ids.to(device), attn_mask.to(device)
            
            logits = model(images, input_ids, attn_mask)
            
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), input_ids.view(B*T), ignore_index=PAD_ID)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            mask = input_ids != PAD_ID
            correct = ((preds == input_ids) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
    
    avg_val_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_tokens
    wandb.log({"val_loss": avg_val_loss, "val_acc": accuracy})
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}  Val Loss={avg_val_loss:.4f}  Val Acc={accuracy:.4f}")
    
    # Visual inspection: show one image + decoded prediction
    test_img = to_pil_image(images[0].cpu())
    decoded_pred = decode_caption(preds[0].unsqueeze(0))
    decoded_true = decode_caption(input_ids[0].unsqueeze(0))
    wandb.log({
        "sample_image": wandb.Image(test_img, caption=f"Pred: {decoded_pred[0]}\nTrue: {decoded_true[0]}")
    })
    
    # Update learning rate
    scheduler.step()
    print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

# Save final model
model_save_path = "mmtransformer_final.pth"
torch.save(model.cpu().state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
model.to(device)  # Move back to device if needed for further use