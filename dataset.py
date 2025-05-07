#dataset.py
import torch
from transformers import CLIPProcessor, CLIPTokenizer
from datasets import load_dataset

dataset = load_dataset("nlphuji/flickr30k")  # huggingface dataset

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ Add PAD token safely
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
PAD_ID = tokenizer.pad_token_id  
MAX_LEN=77


def collate_fn(samples):
    all_images = []
    all_input_ids = []
    all_attn_masks = []

    for sample in samples:
        img = sample["image"]
        captions = sample["caption"][:5]

        for cap in captions:
            all_images.append(img)

            # Tokenize manually
            tokens = tokenizer.encode(cap, add_special_tokens=False)
            tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

            if len(tokens) > MAX_LEN:
                tokens = tokens[:MAX_LEN]

            pad_len = MAX_LEN - len(tokens)
            padding = [tokenizer.pad_token_id] * pad_len  # Usually = 0
            input_ids = tokens + padding
            attention_mask = [1] * len(tokens) + [0] * pad_len

            all_input_ids.append(input_ids)
            all_attn_masks.append(attention_mask)

    pixel_values = processor(images=all_images, return_tensors="pt")["pixel_values"]
    input_ids_tensor = torch.tensor(all_input_ids)
    attn_mask_tensor = torch.tensor(all_attn_masks)

    return pixel_values, input_ids_tensor, attn_mask_tensor



def get_flickr30k_splits(train_ratio=0.9, seed=42):
    """
    Load Flickr30k test split and create train/val subsets.
    Returns:
        train_ds, val_ds: HuggingFace Dataset objects
    """
    full_ds = load_dataset("nlphuji/flickr30k", split="test")
    train_ds, val_ds = full_ds.train_test_split(
        test_size=1 - train_ratio, seed=seed).values()
    return train_ds, val_ds
