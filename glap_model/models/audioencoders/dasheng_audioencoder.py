#!/usr/bin/env python3
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from dasheng import dasheng_base


def parse_spectransforms(transforms: Union[List, Dict]):
    import torchaudio.transforms as audio_transforms

    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    if isinstance(transforms, dict):
        return torch.nn.Sequential(
            *[
                getattr(audio_transforms, trans_name)(**v)
                for trans_name, v in transforms.items()
            ]
        )
    elif isinstance(transforms, list):
        return torch.nn.Sequential(
            *[
                getattr(audio_transforms, trans_name)(**v)
                for item in transforms
                for trans_name, v in item.items()
            ]
        )
    else:
        raise ValueError("Transform unknown")


class DashengWrapper(nn.Module):
    def __init__(self, *args, pretrained_from: Optional[str] = None, **kwargs):
        super().__init__()
        spectransforms = None
        if "spectransforms" in kwargs:
            spectransforms = parse_spectransforms(kwargs.pop("spectransforms"))

        self.model = dasheng_base(*args, **kwargs, spectransforms=spectransforms)
        self.embed_dim = self.model.embed_dim
        # Just remove the last output layer
        if pretrained_from is not None:
            print(f"Load pretrained audio encoder model from {pretrained_from}")
            dump = torch.load(pretrained_from, map_location="cpu")
            self.model.load_state_dict(dump["model"], strict=False)
        self.model.outputlayer = torch.nn.Identity()

    def forward(
        self, input: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute spectrogram
        spec = self.model.forward_to_spec(input)

        # Pad spectrogram time dim to next multiple of target_length
        target_length = self.model.target_length
        if spec.shape[-1] > target_length:
            remainder = spec.shape[-1] % target_length
            if remainder != 0:
                pad_amount = target_length - remainder
                spec = torch.nn.functional.pad(spec, (0, pad_amount))

        # Process through the model's spectrogram pipeline (patch embed, split, features, cat)
        full_output = self.model.forward_spectrogram(spec)

        # Each chunk contributes target_length_in_patches positions along dim=1
        chunk_size_in_patches = target_length // self.model.patch_stride[-1]
        # Split into chunks, mean each chunk, then mean across chunks
        chunks = full_output.split(chunk_size_in_patches, dim=1)
        chunk_means = [c.mean(1) for c in chunks]
        emb = torch.stack(chunk_means).mean(0)
        return emb


__all__ = [DashengWrapper]
