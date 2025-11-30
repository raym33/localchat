"""Configuration settings for the Dia2 conversation app."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Dia2 Model Settings
    dia2_model: str = "nari-labs/Dia2-2B"
    dia2_device: str = "cuda"  # cuda or cpu
    dia2_dtype: str = "bfloat16"

    # Generation settings
    cfg_scale: float = 3.0
    temperature: float = 0.8
    top_k: int = 50
    use_cuda_graph: bool = True

    # LM Studio local server for conversation AI
    lm_studio_url: str = "http://10.183.140.67:1234"
    lm_studio_model: str = "qwen3-4b-thinking-2507"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Audio settings
    sample_rate: int = 24000

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            dia2_model=os.getenv("DIA2_MODEL", "nari-labs/Dia2-2B"),
            dia2_device=os.getenv("DIA2_DEVICE", "cuda"),
            dia2_dtype=os.getenv("DIA2_DTYPE", "bfloat16"),
            cfg_scale=float(os.getenv("DIA2_CFG_SCALE", "3.0")),
            temperature=float(os.getenv("DIA2_TEMPERATURE", "0.8")),
            lm_studio_url=os.getenv("LM_STUDIO_URL", "http://10.183.140.67:1234"),
            lm_studio_model=os.getenv("LM_STUDIO_MODEL", "qwen3-4b-thinking-2507"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )


config = Config.from_env()
