# file: save_embeddings.py

import logging
import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import psutil
import GPUtil

os.makedirs('embeddings', exist_ok=True)

# Logging and Debugging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_system_info():
    """
    Print detailed system and GPU information
    """
    # CPU Information
    logger.info(
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    logger.info(
        f"Total CPU Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # GPU Information
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"GPU Name: {gpu.name}")
            logger.info(f"GPU Total Memory: {gpu.memoryTotal} MB")
            logger.info(f"GPU Free Memory: {gpu.memoryFree} MB")
            logger.info(f"GPU Used Memory: {gpu.memoryUsed} MB")
    except Exception as e:
        logger.warning(f"Could not retrieve GPU information: {e}")


def print_model_info(model, tokenizer):
    """
    Print detailed model configuration
    """
    logger.info("Model Configuration:")
    logger.info(
        f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Vocabulary Size: {len(tokenizer)}")
    logger.info(f"Pad Token: {tokenizer.pad_token}")
    logger.info(f"Pad Token ID: {tokenizer.pad_token_id}")


# Constants
MAX_SEQ_LEN = 77
DATASET_SPLIT = 'test'

print_system_info()
# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load models
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Add pad token if not present
new_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if new_tokens > 0:
    logger.info("Resizing model for newly added tokens")
    # Resize text embedding layer manually for CLIPModel
    old_emb = model.text_model.embeddings.token_embedding
    new_emb = torch.nn.Embedding(len(tokenizer), old_emb.embedding_dim)

    # Copy existing weights
    new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data

    # Assign the new embedding layer
    model.text_model.embeddings.token_embedding = new_emb.to(
        old_emb.weight.device)

# ✅ Update pad_token_id here
pad_token_id = tokenizer.pad_token_id
# Print model information
print_model_info(model, tokenizer)
# Dataset
logger.info("Loading dataset")
dataset = load_dataset("nlphuji/flickr30k")[DATASET_SPLIT]
logger.info(f"Dataset size: {len(dataset)} samples")

# Store arrays
image_embeddings = []
text_embeddings = []
img_ids = []
caption_data = []
combined_embeddings = []
sample_dicts=[]

# Projection for image features [768 → 512]
image_proj = torch.nn.Linear(768, 512).to(device)


@torch.no_grad()
def process_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    logger.debug(f"Image input shape: {inputs['pixel_values'].shape}")
    out = model.vision_model(**inputs)
    patches = out.last_hidden_state[:, 1:, :]  # remove CLS
    processed_img = image_proj(patches).squeeze(0).to(device)  # [49, 512]
    logger.debug(f"Processed image embedding shape: {processed_img.shape}")
    return processed_img


@torch.no_grad()
def process_caption(caption, pad_token_id):
    input_ids = tokenizer.encode(
        caption,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=True
    )
    logger.debug(f"Original input IDs length: {len(input_ids)}")
    pad_len = MAX_SEQ_LEN - len(input_ids)
    input_ids += [pad_token_id] * pad_len
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    logger.debug(f"Input tensor shape: {input_tensor.shape}")
    text_emb = model.text_model.embeddings(input_ids=input_tensor).squeeze(0)
    logger.debug(f"Processed text embedding shape: {text_emb.shape}")
    # [77, 512]
    return text_emb


# Processing loop with enhanced logging
total_processed = 0
skipped_samples = 0

# Iterate and save
for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing")):
    try:
        # Detailed sample logging

        logger.debug(f"Sample details: {sample.keys()}")
        caption = sample['caption'][0]
        img_id = sample['img_id']
        logger.debug(f"Processing sample {sample_idx}, image ID: {img_id}")
        logger.debug(f"Caption: {caption}")
        # Process image on GPU
        img_emb = process_image(sample['image'])             # [49, 512]
        # # Process first caption on GPU 77,512
        txt_emb = process_caption(sample['caption'][0], pad_token_id)
        # Validate embedding shape with detailed error reporting

        def validate_shape(emb, expected_shape, name):
            if emb.shape != expected_shape:
                logger.error(
                    f"{name} embedding shape mismatch. Expected {expected_shape}, Got {emb.shape}")
                raise ValueError(f"{name} embedding shape mismatch")

        validate_shape(img_emb, (49, 512), "Image")
        validate_shape(txt_emb, (77, 512), "Text")

        
        try:
            combined = torch.cat([img_emb, txt_emb], dim=0)
            validate_shape(combined, (126, 512), "Combined")
            combined_embeddings.append(combined.cpu())
        except ValueError as ve:
            logger.error(f"Shape error on sample {sample_idx}: {ve}")
            skipped_samples += 1
            continue
        image_embeddings.append(img_emb.cpu())
        text_embeddings.append(txt_emb.cpu())
        img_ids.append(sample['img_id'])
        # Prepre token ids
        token_ids = tokenizer.encode(
            caption,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True
        )
        token_ids = token_ids[:MAX_SEQ_LEN]
        token_ids += [pad_token_id] * (MAX_SEQ_LEN - len(token_ids))

        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

        caption_data.append({
            "img_id": img_id,
            "caption": caption,
            "tokens": token_ids,
            "tokens_str": token_strs
        })
        total_processed += 1
        # Periodic status update
        if total_processed % 100 == 0:
            logger.info(f"Processed {total_processed} samples")
            logger.info(
                f"Current memory usage: {psutil.virtual_memory().percent}%")
    except Exception as e:
        logger.error(f"Skipped sample {sample_idx} due to error: {e}")
        skipped_samples += 1
        continue

# Final logging and saving
logger.info(f"Total samples processed: {total_processed}")
logger.info(f"Total samples skipped: {skipped_samples}")
# Save as PyTorch (.pt) files with error handling
try:
    torch.save(torch.stack(image_embeddings), "embeddings/image_embeddings.pt")
    logger.info("Saved image embeddings")

    torch.save(torch.stack(text_embeddings), "embeddings/text_embeddings.pt")
    logger.info("Saved text embeddings")

    torch.save(torch.stack(combined_embeddings),
               "embeddings/combined_embeddings.pt")
    logger.info("Saved combined embeddings")

    torch.save(torch.tensor(img_ids), "embeddings/img_ids.pt")
    logger.info("Saved image IDs")
    torch.save(torch.tensor(
        [entry["tokens"] for entry in caption_data]), "embeddings/caption_targets.pt")
    logger.info("Saved caption targets")

except Exception as e:
    logger.error(f"Error saving embeddings: {e}")

# Save tokenizer
tokenizer.save_pretrained("embeddings/tokenizer/")
logger.info("Saved tokenizer")

# Save captions JSON
try:
    with open("embeddings/captions_tokenized.json", "w", encoding="utf-8") as f:
        json.dump(caption_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved captions JSON")
except Exception as e:
    logger.error(f"Error saving captions JSON: {e}")

# Final summary
logger.info("✅ Embedding generation complete")
logger.info(f"✅ Concatenated Shape: {combined.shape}")
