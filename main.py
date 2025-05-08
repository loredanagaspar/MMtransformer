# main.py

from clip_utils import load_clip_with_pad_token

tokenizer, clip_model = load_clip_with_pad_token()
tokenizer = CLIPTokenizer.from_pretrained("tokenizer/")
# Use globally
print("PAD token ID:", tokenizer.pad_token_id)
print("Vocab size:", len(tokenizer))
