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

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def decode_caption(input_ids):
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


# Config
VOCAB_SIZE = tokenizer.vocab_size
BATCH_SIZE = 5
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LR_DECAY = 0.9
PAD_ID = 0
EMBED_DIM = 512
MAX_SEQ_LEN = 77

# Wandb init
wandb.init(project="clip-captioning")

# Load dataset
train_ds, val_ds = get_flickr30k_splits()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MMTransformer(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM).to(device)
optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

# Train Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, input_ids, attn_mask in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        images, input_ids, attn_mask = images.to(
            device), input_ids.to(device), attn_mask.to(device)
        logits = model(images, input_ids, attn_mask)

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(
            B*T, V), input_ids.view(B*T), ignore_index=PAD_ID)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss})

    # Eval
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for images, input_ids, attn_mask in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            images, input_ids, attn_mask = images.to(
                device), input_ids.to(device), attn_mask.to(device)
            logits = model(images, input_ids, attn_mask)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(
                B*T, V), input_ids.view(B*T), ignore_index=PAD_ID)
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

    scheduler.step()

# Save final
torch.save(model.cpu().state_dict(), "mmtransformer_final.pth")
model.to(device)
