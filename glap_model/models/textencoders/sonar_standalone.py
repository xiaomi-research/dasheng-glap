"""
Standalone SONAR text encoder - pure PyTorch implementation.

Replaces sonar-space and fairseq2 dependencies with a self-contained
24-layer Transformer encoder + NLLB tokenizer using only sentencepiece.
"""

import math
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Checkpoint URLs and defaults
# ============================================================================

# Default cache directory for SONAR model files
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sonar_standalone"

# Tokenizer URL (small file, auto-downloaded)
SONAR_TOKENIZER_URL = (
    "https://dl.fbaipublicfiles.com/SONAR/sentencepiece.source.256000.model"
)

# Supported NLLB languages (200+)
NLLB_LANGUAGES = [
    "ace_Arab",
    "ace_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "afr_Latn",
    "ajp_Arab",
    "aka_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "asm_Beng",
    "ast_Latn",
    "awa_Deva",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "ben_Beng",
    "bho_Deva",
    "bjn_Arab",
    "bjn_Latn",
    "bod_Tibt",
    "bos_Latn",
    "bug_Latn",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cjk_Latn",
    "ckb_Arab",
    "crh_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "dik_Latn",
    "dyu_Latn",
    "dzo_Tibt",
    "ell_Grek",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "ewe_Latn",
    "fao_Latn",
    "pes_Arab",
    "fij_Latn",
    "fin_Latn",
    "fon_Latn",
    "fra_Latn",
    "fur_Latn",
    "fuv_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "grn_Latn",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hne_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "kat_Geor",
    "knc_Arab",
    "knc_Latn",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kon_Latn",
    "kor_Hang",
    "kmr_Latn",
    "lao_Laoo",
    "lvs_Latn",
    "lij_Latn",
    "lim_Latn",
    "lin_Latn",
    "lit_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "ltz_Latn",
    "lua_Latn",
    "lug_Latn",
    "luo_Latn",
    "lus_Latn",
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "min_Latn",
    "mkd_Cyrl",
    "plt_Latn",
    "mlt_Latn",
    "mni_Beng",
    "khk_Cyrl",
    "mos_Latn",
    "mri_Latn",
    "zsm_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "npi_Deva",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "gaz_Latn",
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
    "pap_Latn",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
    "pbt_Arab",
    "quy_Latn",
    "ron_Latn",
    "run_Latn",
    "rus_Cyrl",
    "sag_Latn",
    "san_Deva",
    "sat_Beng",
    "scn_Latn",
    "shn_Mymr",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "als_Latn",
    "srd_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "szl_Latn",
    "tam_Taml",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
    "tir_Ethi",
    "taq_Latn",
    "taq_Tfng",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "tzm_Tfng",
    "uig_Arab",
    "ukr_Cyrl",
    "umb_Latn",
    "urd_Arab",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "yue_Hant",
    "zho_Hans",
    "zho_Hant",
    "zul_Latn",
]

# Pre-computed language -> SONAR-vocab token-ID mapping.
# Extracted from fairseq2's NllbTokenizer (uses ICU collation, which sorts
# some scripts differently from Python's default str.sort).  Do NOT derive
# these dynamically; the ordering is fixed by the checkpoint.
NLLB_LANG_TOKEN_IDS = {
    "ace_Arab": 256001,
    "ace_Latn": 256002,
    "acm_Arab": 256003,
    "acq_Arab": 256004,
    "aeb_Arab": 256005,
    "afr_Latn": 256006,
    "ajp_Arab": 256007,
    "aka_Latn": 256008,
    "amh_Ethi": 256009,
    "apc_Arab": 256010,
    "arb_Arab": 256011,
    "ars_Arab": 256012,
    "ary_Arab": 256013,
    "arz_Arab": 256014,
    "asm_Beng": 256015,
    "ast_Latn": 256016,
    "awa_Deva": 256017,
    "ayr_Latn": 256018,
    "azb_Arab": 256019,
    "azj_Latn": 256020,
    "bak_Cyrl": 256021,
    "bam_Latn": 256022,
    "ban_Latn": 256023,
    "bel_Cyrl": 256024,
    "bem_Latn": 256025,
    "ben_Beng": 256026,
    "bho_Deva": 256027,
    "bjn_Arab": 256028,
    "bjn_Latn": 256029,
    "bod_Tibt": 256030,
    "bos_Latn": 256031,
    "bug_Latn": 256032,
    "bul_Cyrl": 256033,
    "cat_Latn": 256034,
    "ceb_Latn": 256035,
    "ces_Latn": 256036,
    "cjk_Latn": 256037,
    "ckb_Arab": 256038,
    "crh_Latn": 256039,
    "cym_Latn": 256040,
    "dan_Latn": 256041,
    "deu_Latn": 256042,
    "dik_Latn": 256043,
    "dyu_Latn": 256044,
    "dzo_Tibt": 256045,
    "ell_Grek": 256046,
    "eng_Latn": 256047,
    "epo_Latn": 256048,
    "est_Latn": 256049,
    "eus_Latn": 256050,
    "ewe_Latn": 256051,
    "fao_Latn": 256052,
    "pes_Arab": 256053,
    "fij_Latn": 256054,
    "fin_Latn": 256055,
    "fon_Latn": 256056,
    "fra_Latn": 256057,
    "fur_Latn": 256058,
    "fuv_Latn": 256059,
    "gla_Latn": 256060,
    "gle_Latn": 256061,
    "glg_Latn": 256062,
    "grn_Latn": 256063,
    "guj_Gujr": 256064,
    "hat_Latn": 256065,
    "hau_Latn": 256066,
    "heb_Hebr": 256067,
    "hin_Deva": 256068,
    "hne_Deva": 256069,
    "hrv_Latn": 256070,
    "hun_Latn": 256071,
    "hye_Armn": 256072,
    "ibo_Latn": 256073,
    "ilo_Latn": 256074,
    "ind_Latn": 256075,
    "isl_Latn": 256076,
    "ita_Latn": 256077,
    "jav_Latn": 256078,
    "jpn_Jpan": 256079,
    "kab_Latn": 256080,
    "kac_Latn": 256081,
    "kam_Latn": 256082,
    "kan_Knda": 256083,
    "kas_Arab": 256084,
    "kas_Deva": 256085,
    "kat_Geor": 256086,
    "knc_Arab": 256087,
    "knc_Latn": 256088,
    "kaz_Cyrl": 256089,
    "kbp_Latn": 256090,
    "kea_Latn": 256091,
    "khm_Khmr": 256092,
    "kik_Latn": 256093,
    "kin_Latn": 256094,
    "kir_Cyrl": 256095,
    "kmb_Latn": 256096,
    "kon_Latn": 256097,
    "kor_Hang": 256098,
    "kmr_Latn": 256099,
    "lao_Laoo": 256100,
    "lvs_Latn": 256101,
    "lij_Latn": 256102,
    "lim_Latn": 256103,
    "lin_Latn": 256104,
    "lit_Latn": 256105,
    "lmo_Latn": 256106,
    "ltg_Latn": 256107,
    "ltz_Latn": 256108,
    "lua_Latn": 256109,
    "lug_Latn": 256110,
    "luo_Latn": 256111,
    "lus_Latn": 256112,
    "mag_Deva": 256113,
    "mai_Deva": 256114,
    "mal_Mlym": 256115,
    "mar_Deva": 256116,
    "min_Latn": 256117,
    "mkd_Cyrl": 256118,
    "plt_Latn": 256119,
    "mlt_Latn": 256120,
    "mni_Beng": 256121,
    "khk_Cyrl": 256122,
    "mos_Latn": 256123,
    "mri_Latn": 256124,
    "zsm_Latn": 256125,
    "mya_Mymr": 256126,
    "nld_Latn": 256127,
    "nno_Latn": 256128,
    "nob_Latn": 256129,
    "npi_Deva": 256130,
    "nso_Latn": 256131,
    "nus_Latn": 256132,
    "nya_Latn": 256133,
    "oci_Latn": 256134,
    "gaz_Latn": 256135,
    "ory_Orya": 256136,
    "pag_Latn": 256137,
    "pan_Guru": 256138,
    "pap_Latn": 256139,
    "pol_Latn": 256140,
    "por_Latn": 256141,
    "prs_Arab": 256142,
    "pbt_Arab": 256143,
    "quy_Latn": 256144,
    "ron_Latn": 256145,
    "run_Latn": 256146,
    "rus_Cyrl": 256147,
    "sag_Latn": 256148,
    "san_Deva": 256149,
    "sat_Beng": 256150,
    "scn_Latn": 256151,
    "shn_Mymr": 256152,
    "sin_Sinh": 256153,
    "slk_Latn": 256154,
    "slv_Latn": 256155,
    "smo_Latn": 256156,
    "sna_Latn": 256157,
    "snd_Arab": 256158,
    "som_Latn": 256159,
    "sot_Latn": 256160,
    "spa_Latn": 256161,
    "als_Latn": 256162,
    "srd_Latn": 256163,
    "srp_Cyrl": 256164,
    "ssw_Latn": 256165,
    "sun_Latn": 256166,
    "swe_Latn": 256167,
    "swh_Latn": 256168,
    "szl_Latn": 256169,
    "tam_Taml": 256170,
    "tat_Cyrl": 256171,
    "tel_Telu": 256172,
    "tgk_Cyrl": 256173,
    "tgl_Latn": 256174,
    "tha_Thai": 256175,
    "tir_Ethi": 256176,
    "taq_Latn": 256177,
    "taq_Tfng": 256178,
    "tpi_Latn": 256179,
    "tsn_Latn": 256180,
    "tso_Latn": 256181,
    "tuk_Latn": 256182,
    "tum_Latn": 256183,
    "tur_Latn": 256184,
    "twi_Latn": 256185,
    "tzm_Tfng": 256186,
    "uig_Arab": 256187,
    "ukr_Cyrl": 256188,
    "umb_Latn": 256189,
    "urd_Arab": 256190,
    "uzn_Latn": 256191,
    "vec_Latn": 256192,
    "vie_Latn": 256193,
    "war_Latn": 256194,
    "wol_Latn": 256195,
    "xho_Latn": 256196,
    "ydd_Hebr": 256197,
    "yor_Latn": 256198,
    "yue_Hant": 256199,
    "zho_Hans": 256200,
    "zho_Hant": 256201,
    "zul_Latn": 256202,
}


# ============================================================================
# Model Architecture
# ============================================================================


class SinusoidalPositionEncoder(nn.Module):
    """Fixed sinusoidal positional encoding with legacy fairseq compatibility."""

    def __init__(self, encoding_dim: int, max_seq_len: int, _legacy_pad_idx: int = 1):
        super().__init__()
        assert encoding_dim % 2 == 0
        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len
        # Legacy offset for fairseq compatibility: offset = 1 + pad_idx
        start_step = 1 + _legacy_pad_idx
        steps = torch.arange(start_step, start_step + max_seq_len, dtype=torch.float32)
        self.register_buffer(
            "freqs", self._build_freqs(steps, encoding_dim), persistent=False
        )

    @staticmethod
    def _build_freqs(steps: Tensor, encoding_dim: int) -> Tensor:
        num_sin = encoding_dim // 2
        indices = torch.arange(num_sin, dtype=torch.float32)
        freq_vals = torch.exp(indices * -math.log(10000.0) / (num_sin - 1))
        # (S, E/2)
        l_half = torch.outer(steps, freq_vals)
        r_half = l_half[:, : encoding_dim - num_sin].clone()
        return torch.cat([l_half.sin(), r_half.cos()], dim=-1)

    def forward(self, seqs: Tensor) -> Tensor:
        seq_len = seqs.size(-2)
        return (seqs.float() + self.freqs[:seq_len]).type_as(seqs)


class MultiheadAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, model_dim: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0

        self.q_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout_p = dropout_p

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seq_len, _ = queries.shape

        # Project Q, K, V
        q = (
            self.q_proj(queries)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(keys)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(values)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply padding mask
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )

        # Softmax in float32 for numerical stability
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        if self.training and self.attn_dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p)

        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.model_dim)
        return self.output_proj(attn)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with PRE-norm ordering.
    Attribute names match fairseq2 checkpoint keys: self_attn_layer_norm, ffn_layer_norm, etc.
    """

    def __init__(
        self, model_dim: int, num_heads: int, ffn_inner_dim: int, dropout_p: float = 0.1
    ):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(model_dim)
        self.self_attn = MultiheadAttention(model_dim, num_heads, dropout_p=dropout_p)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)
        # FFN matches fairseq2 naming: inner_proj, output_proj
        self.ffn = _FeedForwardNetwork(model_dim, ffn_inner_dim, dropout_p)

    def forward(self, seqs: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm self-attention + residual
        residual = seqs
        seqs = self.self_attn_layer_norm(seqs)
        seqs = self.self_attn(seqs, seqs, seqs, padding_mask)
        seqs = seqs + residual

        # Pre-norm FFN + residual
        residual = seqs
        seqs = self.ffn_layer_norm(seqs)
        seqs = self.ffn(seqs)
        seqs = seqs + residual
        return seqs


class _FeedForwardNetwork(nn.Module):
    """FFN with inner_proj/output_proj naming to match fairseq2 checkpoint."""

    def __init__(self, model_dim: int, inner_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.inner_proj = nn.Linear(model_dim, inner_dim, bias=True)
        self.output_proj = nn.Linear(inner_dim, model_dim, bias=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inner_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return x


class _TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers. Named 'encoder' to match checkpoint keys."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, seqs: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            seqs = layer(seqs, padding_mask)
        return seqs


class _EmbeddingFrontend(nn.Module):
    """
    Embedding frontend with scaled embedding + sinusoidal PE + dropout.
    Named 'encoder_frontend' to match checkpoint keys.
    """

    def __init__(
        self,
        embed: nn.Embedding,
        pos_encoder: SinusoidalPositionEncoder,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.embed = embed
        self.pos_encoder = pos_encoder
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, token_ids: Tensor) -> Tensor:
        seqs = self.embed(token_ids)
        # Scale by sqrt(model_dim) as in the original Transformer paper
        seqs = seqs * math.sqrt(seqs.size(-1))
        seqs = self.pos_encoder(seqs)
        seqs = self.dropout(seqs)
        return seqs


class SonarTextEncoder(nn.Module):
    """
    SONAR text encoder: 24-layer Transformer encoder with sinusoidal PE
    and mean pooling. Compatible with the official SONAR checkpoint.

    The module structure matches fairseq2's SonarTextTransformerEncoderModel:
      - encoder_frontend: embedding + sinusoidal PE + dropout
      - encoder: 24 Transformer layers (each with self_attn_layer_norm, self_attn,
                 ffn_layer_norm, ffn with inner_proj/output_proj)
      - layer_norm: final LayerNorm on encoder output
    """

    def __init__(
        self,
        vocab_size: int = 256206,
        model_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        ffn_inner_dim: int = 8192,
        max_seq_len: int = 514,
        pad_idx: int = 0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pad_idx = pad_idx

        # Embedding frontend (matches checkpoint key: encoder_frontend.*)
        embed = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        # _legacy_pad_idx=1 matches the fairseq2 SONAR checkpoint
        # (trained with that offset in the sinusoidal PE).
        pos_encoder = SinusoidalPositionEncoder(
            model_dim, max_seq_len, _legacy_pad_idx=1
        )
        self.encoder_frontend = _EmbeddingFrontend(embed, pos_encoder, dropout_p)

        # Encoder layers (matches checkpoint key: encoder.layers.{i}.*)
        layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, num_heads, ffn_inner_dim, dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.encoder = _TransformerEncoder(layers)

        # Final layer norm (matches checkpoint key: layer_norm.*)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        token_ids: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            token_ids: (B, S) integer token IDs
            padding_mask: (B, S) boolean mask where True = padded position

        Returns:
            sentence_embeddings: (B, model_dim) mean-pooled embeddings
        """
        # Frontend
        seqs = self.encoder_frontend(token_ids)

        # Encoder
        seqs = self.encoder(seqs, padding_mask)

        # Final layer norm
        seqs = self.layer_norm(seqs)

        # Mean pooling (matching fairseq2 Pooling.MEAN)
        if padding_mask is None:
            sentence_embeddings = seqs.sum(dim=1) / (seqs.size(1) + 1e-7)
        else:
            mask = (~padding_mask).unsqueeze(-1).float()
            seqs = seqs * mask
            lengths = mask.sum(dim=1).clamp(min=1e-7)
            sentence_embeddings = seqs.sum(dim=1) / lengths

        return sentence_embeddings


# ============================================================================
# Tokenizer
# ============================================================================


class NllbTokenizer:
    """
    Standalone NLLB tokenizer using sentencepiece directly.
    Supports 200+ languages via language-specific control symbols.

    The SONAR/fairseq2 vocabulary extends the base 256000-token sentencepiece
    model with 206 control symbols (202 language tokens + 3 data-source
    markers + 1 pad).  Content token IDs from the base spm model are shifted
    by +1 (PAD occupies index 0, pushing everything else up by one).

    Token layout:
        0         = <pad>
        1         = <unk>
        2         = <s>   (BOS)
        3         = </s>  (EOS)
        4..255999 = content tokens  (spm_id + 1)
        256000    = pad-placeholder (original spm pad, shifted)
        256001..256202 = language tokens
        256203..256205 = <MINED_DATA>, <MMT_BT_DATA>, <SMT_BT_DATA>
    """

    def __init__(self, model_path: str | Path, langs: Optional[List[str]] = None):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece is required: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(str(model_path)):
            raise RuntimeError(f"Failed to load SentencePiece model from {model_path}")

        if langs is None:
            langs = NLLB_LANGUAGES

        self.langs = set(langs)

        # The SONAR/NLLB vocab has special tokens:
        # After the SONAR reordering: PAD=0, UNK=1, BOS=2, EOS=3
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        # Use the pre-computed mapping extracted from fairseq2.
        # The fairseq2 NllbTokenizer uses ICU collation which sorts some
        # scripts differently from Python's default str.sort(), so we
        # cannot derive the IDs dynamically.
        self._lang_token_to_idx = NLLB_LANG_TOKEN_IDS

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 206

    def create_encoder(self, lang: str = "eng_Latn"):
        """Create an encoding function for a specific language."""
        if lang not in self.langs:
            raise ValueError(f"Unsupported language: {lang}")

        lang_idx = self._lang_token_to_idx.get(lang)
        eos_idx = self.eos_idx

        def encode(text: str) -> List[int]:
            """Tokenize text with language prefix and EOS suffix.

            Produces the same token IDs as fairseq2's NllbTokenizer:
                [lang_token_id, spm_tok+1, spm_tok+1, ..., </s>]
            """
            spm_ids = self.sp.encode(text, out_type=int)

            # Apply the +1 offset that fairseq2 uses (PAD at 0 shifted everything)
            content_ids = [tid + 1 for tid in spm_ids]

            if lang_idx is not None:
                token_ids = [lang_idx] + content_ids
            else:
                token_ids = content_ids
            token_ids.append(eos_idx)
            return token_ids

        return encode


# ============================================================================
# Checkpoint Management
# ============================================================================


def _download_file(url: str, dest: Path, desc: str = "file") -> Path:
    """Download a file with progress reporting."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".tmp")
    print(f"Downloading {desc} from {url}...", flush=True)

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(tmp_dest), reporthook=progress_hook)
        print(f"\n  Saved to {dest}", flush=True)
        tmp_dest.rename(dest)
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
    return dest


def ensure_checkpoint(cache_dir: Optional[str | Path] = None) -> Path:
    """Return cached SONAR text encoder checkpoint path.

    The checkpoint is not downloaded automatically — use ``glap_inference()``
    which loads the full GLAP checkpoint (includes SONAR weights).  If you
    need the standalone SONAR encoder, download the checkpoint manually::

        mkdir -p ~/.cache/sonar_standalone
        wget -O ~/.cache/sonar_standalone/sonar_text_encoder.pt \\
            https://dl.fbaipublicfiles.com/SONAR/sonar_text_encoder.pt
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    dest = cache_dir / "sonar_text_encoder.pt"
    if not dest.exists():
        raise FileNotFoundError(
            f"SONAR text encoder checkpoint not found at {dest}. "
            f"Download it manually or use glap_inference() which loads "
            f"the full GLAP checkpoint with SONAR weights included."
        )
    return dest


def ensure_tokenizer(cache_dir: Optional[str | Path] = None) -> Path:
    """Download and cache the SONAR/NLLB tokenizer model (small, ~5MB)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return _download_file(
        SONAR_TOKENIZER_URL,
        cache_dir / "sentencepiece.source.256000.model",
        "SONAR tokenizer",
    )


def load_sonar_encoder(
    checkpoint_path: Optional[str | Path] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    cache_dir: Optional[str | Path] = None,
) -> SonarTextEncoder:
    """
    Load the SONAR text encoder with pretrained weights.

    For standalone use, the SONAR checkpoint must be cached at
    ``~/.cache/sonar_standalone/sonar_text_encoder.pt`` (no auto-download).
    When using ``glap_inference()``, the GLAP checkpoint already includes
    these weights.

    Args:
        checkpoint_path: Path to SONAR checkpoint. If None, downloads automatically.
        device: Device to load the model on.
        dtype: Data type for the model.
        cache_dir: Directory for caching downloads.

    Returns:
        SonarTextEncoder with loaded weights.
    """
    if checkpoint_path is None:
        checkpoint_path = ensure_checkpoint(cache_dir)

    # Load checkpoint
    ckpt = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Build model with matching architecture
    model = SonarTextEncoder(
        vocab_size=256206,
        model_dim=1024,
        num_layers=24,
        num_heads=16,
        ffn_inner_dim=8192,
        max_seq_len=514,
        pad_idx=0,
        dropout_p=0.1,
    )

    # Load weights (strict=True to verify architecture matches)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(dtype=dtype, device=device)
    model.eval()
    return model


def load_tokenizer(
    tokenizer_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> NllbTokenizer:
    """Load the NLLB tokenizer."""
    if tokenizer_path is None:
        tokenizer_path = ensure_tokenizer(cache_dir)
    return NllbTokenizer(tokenizer_path)
