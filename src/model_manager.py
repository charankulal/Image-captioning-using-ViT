"""Image Captioning Model Manager - API for easy model integration"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

from src.model import ImageCaptioningModel
from src.vocabulary import Vocabulary


class ImageCaptioningModelManager:
    """Manager class for loading and using the image captioning model"""

    def __init__(self, checkpoint_dir='checkpoints'):
        """Initialize the model manager

        Args:
            checkpoint_dir: Directory containing model checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self, model_name='best_model.pth', vocab_name='vocab.pth'):
        """Load the model and vocabulary from checkpoints

        Args:
            model_name: Name of the model checkpoint file
            vocab_name: Name of the vocabulary file
        """
        model_path = os.path.join(self.checkpoint_dir, model_name)
        vocab_path = os.path.join(self.checkpoint_dir, vocab_name)

        # Load vocabulary - inject Vocabulary class into __main__ namespace
        # This is needed because vocab.pth was saved from a notebook where
        # Vocabulary was defined in __main__
        print(f"Loading vocabulary from {vocab_path}...")

        import sys
        import __main__

        # Inject Vocabulary class into __main__ so pickle can find it
        __main__.Vocabulary = Vocabulary
        sys.modules['__main__'].Vocabulary = Vocabulary

        # Register for safe loading
        torch.serialization.add_safe_globals([Vocabulary])

        try:
            self.vocab = torch.load(vocab_path, weights_only=False)
            print(f"Vocabulary loaded successfully! Size: {len(self.vocab)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary: {str(e)}")

        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint['config']

        # Initialize model
        self.model = ImageCaptioningModel(
            vocab_size=len(self.vocab),
            embed_dim=config['embed_dim'],
            num_decoder_layers=config['num_decoder_layers'],
            num_heads=config['num_heads'],
            forward_expansion=config['forward_expansion'],
            dropout=config['dropout'],
            max_length=config['max_length'],
            pretrained_vit=config['pretrained_vit']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Trained for {checkpoint['epoch'] + 1} epochs")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"Using device: {self.device}")

    def preprocess_image(self, image_path):
        """Preprocess an image for model input

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor [1, 3, 224, 224]
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def generate_caption(self, image_path, method='beam_search', beam_width=5,
                        max_length=50, return_confidence=False):
        """Generate caption for an image

        Args:
            image_path: Path to the image file
            method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search
            max_length: Maximum caption length
            return_confidence: Whether to return confidence score

        Returns:
            caption (str): Generated caption
            confidence (float, optional): Confidence score if return_confidence=True
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model not loaded! Call load() first.")

        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        # Generate caption
        if method == 'beam_search':
            caption_words = self.model.generate_caption_beam_search(
                image_tensor, self.vocab, beam_width=beam_width,
                max_length=max_length, device=self.device
            )
        else:
            caption_words = self.model.generate_caption(
                image_tensor, self.vocab, max_length=max_length, device=self.device
            )

        caption = ' '.join(caption_words)

        if return_confidence:
            # Calculate confidence as average of token probabilities
            confidence = self._calculate_confidence(image_tensor, caption_words)
            return caption, confidence

        return caption

    def _calculate_confidence(self, image_tensor, caption_words):
        """Calculate confidence score for generated caption

        Args:
            image_tensor: Preprocessed image tensor
            caption_words: List of caption words

        Returns:
            Confidence score (0-1)
        """
        if len(caption_words) == 0:
            return 0.0

        with torch.no_grad():
            encoder_out = self.model.encoder(image_tensor)
            caption_indices = [self.vocab.stoi["<SOS>"]] + \
                             [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"])
                              for word in caption_words] + \
                             [self.vocab.stoi["<EOS>"]]

            # Get probabilities for each predicted token
            probs = []
            for i in range(1, len(caption_indices)):
                context = torch.LongTensor(caption_indices[:i]).unsqueeze(0).to(self.device)
                tgt_mask = self.model.decoder.generate_square_subsequent_mask(i, self.device)
                predictions = self.model.decoder(context, encoder_out, tgt_mask)

                # Get probability of the actual token
                token_probs = torch.softmax(predictions[0, -1, :], dim=0)
                token_prob = token_probs[caption_indices[i]].item()
                probs.append(token_prob)

            # Return geometric mean of probabilities
            confidence = np.exp(np.mean(np.log(np.array(probs) + 1e-10)))
            return confidence

    def generate_captions_batch(self, image_paths, method='beam_search', beam_width=5):
        """Generate captions for multiple images

        Args:
            image_paths: List of image paths
            method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search

        Returns:
            List of captions
        """
        captions = []
        for image_path in image_paths:
            caption = self.generate_caption(image_path, method=method, beam_width=beam_width)
            captions.append(caption)
        return captions


def quick_caption(image_path, checkpoint_dir='checkpoints'):
    """Quick one-liner to generate caption for an image

    Args:
        image_path: Path to the image
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Generated caption string
    """
    manager = ImageCaptioningModelManager(checkpoint_dir)
    manager.load()
    return manager.generate_caption(image_path)
