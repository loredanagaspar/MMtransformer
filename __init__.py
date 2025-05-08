# flickr30k_captioning/embed_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from datasets import load_dataset


class Flickr30kSingleCaption(Dataset):
    def __init__(self, clip_model, tokenizer, processor, split='test', max_seq_length=77):
        self.dataset = load_dataset("nlphuji/flickr30k")[split]
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.tokenizer.pad_token_id
        self.image_projection = torch.nn.Linear(768, 512)

    def _process_caption(self, caption: str) -> torch.Tensor:
        ids = self.tokenizer.encode(
            caption,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )
        ids = ids[:self.max_seq_length]
        ids += [self.pad_token_id] * (self.max_seq_length - len(ids))
        return torch.tensor(ids)

    @torch.no_grad()
    def _process_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.clip_model.vision_model(**inputs)
        patch_embeddings = out.last_hidden_state[:, 1:, :]
        return self.image_projection(patch_embeddings).squeeze(0)  # (49, 512)

    @torch.no_grad()
    def embed_text(self, input_ids):
        return self.clip_model.text_model.embeddings(input_ids=input_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        caption = sample['caption'][0]  # Use only the first caption
        image = sample['image']

        input_ids = self._process_caption(caption).unsqueeze(0)
        text_embedding = self.embed_text(
            input_ids=input_ids).squeeze(0)  # (L, 512)
        image_embedding = self._process_image(image)  # (49, 512)

        return {
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "caption": caption,
            "input_ids": input_ids.squeeze(0),
        }


def create_dataloader(clip_model, tokenizer, processor, batch_size=1, shuffle=False):
    dataset = Flickr30kSingleCaption(clip_model, tokenizer, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    new_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if new_tokens > 0:
        old_emb = clip_model.text_model.embeddings.token_embedding
        new_emb = torch.nn.Embedding(len(tokenizer), old_emb.embedding_dim)
        new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
        clip_model.text_model.embeddings.token_embedding = new_emb

    dataloader = create_dataloader(clip_model, tokenizer, processor)

    for batch in dataloader:
        print("📸 Image Embedding:", batch["image_embedding"].shape)
        print("📝 Text Embedding:", batch["text_embedding"].shape)
        print("🧾 Caption:", batch["caption"])
        print("🔤 Tokens:", tokenizer.convert_ids_to_tokens(
            batch["input_ids"][0].tolist()))
        break
