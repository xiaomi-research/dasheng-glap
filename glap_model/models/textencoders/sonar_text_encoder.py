#!/usr/bin/env python3
"""SONAR text encoder wrapper using standalone pure-PyTorch implementation.

Drop-in replacement for the original fairseq2/sonar-based wrapper.
No sonar-space or fairseq2 dependencies required.
"""

import math
from typing import List, Literal, Optional, Sequence

import torch
import torch.nn as nn

from .sonar_standalone import (
    SonarTextEncoder,
    load_tokenizer,
)


class TextEncoderSonarWrapper(nn.Module):
    """
    Wraps the standalone SONAR text encoder with a clean forward interface.

    The interface is compatible with the original GLAP model:
        forward(text_input: Sequence[str], device: torch.device, source_lang: str) -> torch.Tensor
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        # Build model architecture (weights loaded later from GLAP checkpoint)
        self.model = SonarTextEncoder(
            vocab_size=256206,
            model_dim=1024,
            num_layers=24,
            num_heads=16,
            ffn_inner_dim=8192,
            max_seq_len=514,
            pad_idx=0,
            dropout_p=0.1,
        )

        # If an explicit checkpoint_path is given, load it directly
        if checkpoint_path is not None:
            ckpt = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
            )
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=True)

        self.tokenizer = load_tokenizer(
            tokenizer_path=tokenizer_path, cache_dir=cache_dir
        )

        self.embed_dim = self.model.model_dim
        self.max_seq_len = (
            max_seq_len
            if max_seq_len is not None
            else self.model.encoder_frontend.pos_encoder.max_seq_len
        )

    def forward(
        self,
        text_input: Sequence[str],
        device: torch.device,
        source_lang: str = "eng_Latn",
    ) -> torch.Tensor:
        """
        Encode a batch of text strings into sentence embeddings.

        Args:
            text_input: List of text strings to encode.
            device: Target device for the output tensor.
            source_lang: NLLB language code (e.g., "eng_Latn", "zho_Hans").

        Returns:
            Sentence embeddings of shape (batch_size, embed_dim).
        """
        # Create tokenizer encoder for the given language
        tokenizer_encoder = self.tokenizer.create_encoder(lang=source_lang)

        # Tokenize all texts
        all_token_ids: List[List[int]] = []
        for text in text_input:
            token_ids = tokenizer_encoder(text)
            # Truncate to max sequence length
            token_ids = token_ids[: self.max_seq_len]
            all_token_ids.append(token_ids)

        # Find max length for padding
        max_len = max(len(ids) for ids in all_token_ids) if all_token_ids else 0

        # Pad sequences
        batch_size = len(all_token_ids)
        padded_ids = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_idx,
            dtype=torch.long,
            device="cpu",
        )
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device="cpu")

        for i, ids in enumerate(all_token_ids):
            length = len(ids)
            padded_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            # Mark padded positions
            padding_mask[i, length:] = True

        # Run through model
        self.model.eval()
        with torch.no_grad():
            sentence_embeddings = self.model(padded_ids, padding_mask)

        return sentence_embeddings.to(device)
