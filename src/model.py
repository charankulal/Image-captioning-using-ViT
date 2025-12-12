"""Image Captioning Model Architecture"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer decoder"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder using pre-trained ViT"""

    def __init__(self, embed_dim=768, pretrained=True):
        super(VisionTransformerEncoder, self).__init__()

        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            config = ViTConfig(
                hidden_size=embed_dim,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16
            )
            self.vit = ViTModel(config)

        self.embed_dim = embed_dim

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        features = outputs.last_hidden_state
        return features


class TransformerDecoder(nn.Module):
    """Transformer decoder for caption generation"""

    def __init__(self, vocab_size, embed_dim=768, num_layers=6, num_heads=8,
                 forward_expansion=4, dropout=0.1, max_length=100):
        super(TransformerDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_length, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * forward_expansion,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, encoder_out, tgt_mask=None, tgt_padding_mask=None):
        embeddings = self.word_embedding(captions)
        embeddings = self.positional_encoding(embeddings)

        decoder_out = self.transformer_decoder(
            tgt=embeddings,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        predictions = self.fc_out(decoder_out)
        return predictions

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model with ViT encoder and Transformer decoder"""

    def __init__(self, vocab_size, embed_dim=768, num_decoder_layers=6,
                 num_heads=8, forward_expansion=4, dropout=0.1,
                 max_length=100, pretrained_vit=True):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = VisionTransformerEncoder(embed_dim=embed_dim, pretrained=pretrained_vit)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length
        )

    def forward(self, images, captions, tgt_padding_mask=None):
        encoder_out = self.encoder(images)
        caption_len = captions.size(1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(caption_len, captions.device)
        predictions = self.decoder(captions, encoder_out, tgt_mask, tgt_padding_mask)
        return predictions

    def generate_caption(self, image, vocab, max_length=50, device='cuda'):
        """Generate caption for a single image using greedy decoding

        Args:
            image: Preprocessed image tensor [1, 3, 224, 224]
            vocab: Vocabulary object
            max_length: Maximum caption length
            device: Device to use (cuda/cpu)

        Returns:
            List of caption words
        """
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image)
            caption_indices = [vocab.stoi["<SOS>"]]

            for _ in range(max_length):
                caption_tensor = torch.LongTensor(caption_indices).unsqueeze(0).to(device)
                tgt_mask = self.decoder.generate_square_subsequent_mask(len(caption_indices), device)
                predictions = self.decoder(caption_tensor, encoder_out, tgt_mask)
                predicted_id = predictions[0, -1, :].argmax().item()
                caption_indices.append(predicted_id)

                if predicted_id == vocab.stoi["<EOS>"]:
                    break

            caption = [vocab.itos[idx] for idx in caption_indices[1:-1]]
        return caption

    def generate_caption_beam_search(self, image, vocab, beam_width=5, max_length=50, device='cuda'):
        """Generate caption using beam search

        Args:
            image: Preprocessed image tensor [1, 3, 224, 224]
            vocab: Vocabulary object
            beam_width: Number of beams to use
            max_length: Maximum caption length
            device: Device to use (cuda/cpu)

        Returns:
            List of caption words
        """
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image)
            beams = [(0.0, [vocab.stoi["<SOS>"]])]

            for _ in range(max_length):
                new_beams = []

                for score, caption_indices in beams:
                    if caption_indices[-1] == vocab.stoi["<EOS>"]:
                        new_beams.append((score, caption_indices))
                        continue

                    caption_tensor = torch.LongTensor(caption_indices).unsqueeze(0).to(device)
                    tgt_mask = self.decoder.generate_square_subsequent_mask(len(caption_indices), device)
                    predictions = self.decoder(caption_tensor, encoder_out, tgt_mask)

                    log_probs = torch.log_softmax(predictions[0, -1, :], dim=0)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for log_prob, idx in zip(top_log_probs, top_indices):
                        new_score = score + log_prob.item()
                        new_caption = caption_indices + [idx.item()]
                        new_beams.append((new_score, new_caption))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

                if all(caption[-1] == vocab.stoi["<EOS>"] for _, caption in beams):
                    break

            best_caption_indices = beams[0][1]
            caption = [vocab.itos[idx] for idx in best_caption_indices[1:-1]]

        return caption
