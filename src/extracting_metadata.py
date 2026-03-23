# Importing necessary libraries
import warnings
from dataclasses import dataclass
from typing import Optional, List
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import HfApi, ModelCard
warnings.filterwarnings("ignore")
# Initialize API once
api = HfApi()

@dataclass
class ModelMetadata:
    model_id: str
    author: str
    license: Optional[str]
    has_model_card: bool
    architecture: Optional[str]
    num_parameters_estimate: Optional[int]
    num_layers: Optional[int]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    vocab_size: Optional[int]
    max_position_embeddings: Optional[int]
    tokenizer_class: Optional[str]
    tags: List[str]

def get_license(model_id: str) -> Optional[str]:
    """Fetching license from Hugging Face model card metadata."""
    try:
        info = api.model_info(model_id)
        if info.cardData:
            return info.cardData.get("license")
    except Exception:
        pass
    return None


def get_tags(model_id: str) -> List[str]:
    """Fetching tags associated with a model."""
    try:
        info = api.model_info(model_id)
        return list(info.tags or [])
    except Exception:
        return []


def check_model_card(model_id: str) -> bool:
    """Checking if a model card exists."""
    try:
        ModelCard.load(model_id)
        return True
    except Exception:
        return False


def estimate_parameters(config) -> Optional[int]:
    """
    Estimating total parameter count (no weights downloaded).
    Formula:
    embedding layer (vocab x hidden) +
    transformer blocks (12 x hidden^2 x layers)
    """
    hidden = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None))

    layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None))
    vocab = getattr(config, "vocab_size", None)
    if hidden and layers and vocab:
        return vocab * hidden + layers * (12 * hidden * hidden)

    return None

def load_model_meta(model_id: str) -> ModelMetadata:
    """Loading model config + tokenizer and extract metadata."""
    print(f"Loading: {model_id}")

    config = AutoConfig.from_pretrained(model_id)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_class = type(tokenizer).__name__
    except Exception:
        tokenizer_class = None

    hidden_size = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None))

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None))

    num_heads = (
        getattr(config, "num_attention_heads", None)
        or getattr(config, "n_head", None))

    max_ctx = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None))
    metadata = ModelMetadata(
        model_id=model_id,
        author=model_id.split("/")[0] if "/" in model_id else "unknown",
        license=get_license(model_id),
        has_model_card=check_model_card(model_id),
        architecture=config.architectures[0] if config.architectures else None,
        num_parameters_estimate=estimate_parameters(config),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        vocab_size=getattr(config, "vocab_size", None),
        max_position_embeddings=max_ctx,
        tokenizer_class=tokenizer_class,
        tags=get_tags(model_id),
    )
    return metadata