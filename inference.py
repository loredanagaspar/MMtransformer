import torch
from transformers import CLIPProcessor, CLIPTokenizer
from torchvision.transforms import ToTensor
from PIL import Image
import os
from glob import glob

from model import MMTransformer


def greedy_decode(model, image_pil, tokenizer, processor, max_len=77):
    model.eval()
    device = next(model.parameters()).device

    # Prepare image
    inputs = processor(images=image_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1, 3, 224, 224)

    # Start decoding
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    print("🧪 Initial input_ids:", input_ids.tolist())
    print("🧪 Decoded:", tokenizer.decode(input_ids[0]))  # (1, 1)
    attn_mask = torch.ones_like(input_ids)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(pixel_values, input_ids, attn_mask)  # (1, T, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attn_mask = torch.ones_like(input_ids)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    weights_path = "mmtransformer_final.pth"
    image_folder = "sample_images"

    # Load model components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ✅ Add PAD token safely
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    PAD_ID = tokenizer.pad_token_id  
    VOCAB_SIZE = len(tokenizer)
    model = MMTransformer(vocab_size=VOCAB_SIZE)

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process all images in folder
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        caption = greedy_decode(model, image, tokenizer, processor)
        print(f"\n🖼️ {os.path.basename(image_path)} → {caption}")


if __name__ == "__main__":
    main()
