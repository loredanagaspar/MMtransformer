from transformers import CLIPProcessor
from datasets import load_dataset

dataset = load_dataset("nlphuji/flickr30k")  # huggingface dataset


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def collate_fn(samples):
    """
    Custom collate function to expand image across 5 captions.
    Args:
        samples: List of dicts with 'image', 'caption' (list of 5 captions)
    Returns:
        pixel_values: (B*5, 3, 224, 224)
        input_ids: (B*5, 77)
        attention_mask: (B*5, 77)
    """
    all_images = []
    all_captions = []

    for sample in samples:
        img = sample["image"]
        captions = sample["caption"][:5]
        all_images.extend([img] * len(captions))  # duplicate image 5x
        all_captions.extend(captions)

    # Tokenize and process all at once
    inputs = processor(text=all_captions, images=all_images,
                       return_tensors="pt", padding="max_length", truncation=True)
    return inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"]


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
