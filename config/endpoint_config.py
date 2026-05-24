"""Endpoint configuration for local vLLM judges.

Judge panel matches the paper (Sec 4 Experimental Setup):
  MedGemma-4B   (google/medgemma-4b-it)
  BioMistral-7B (BioMistral/BioMistral-7B-DARE)
  Meditron-7B   (epfl-llm/meditron-7b)
  MedAlpaca-7B  (medalpaca/medalpaca-7b)

The panel combines general instruction-tuned models with medically
specialised ones, introducing diversity while remaining fully local
and privacy-preserving.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    """Configuration for one vLLM judge endpoint."""
    id: str
    model: str
    url: str
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    description: Optional[str] = None


# Paper Sec 4: four locally-deployed SLMs spanning three distinct base architectures
DEFAULT_JUDGE_ENDPOINTS = [
    EndpointConfig(
        id="medgemma",
        model="google/medgemma-4b-it",
        url=os.getenv("LOCAL_VLLM_JUDGE1_URL", "http://localhost:8001"),
        description="MedGemma 4B -- Google DeepMind medical fine-tune (sellergren2025medgemma)",
    ),
    EndpointConfig(
        id="biomistral",
        model="BioMistral/BioMistral-7B-DARE",
        url=os.getenv("LOCAL_VLLM_JUDGE2_URL", "http://localhost:8002"),
        description="BioMistral 7B-DARE -- biomedical corpus fine-tune (labrak2024biomistral)",
    ),
    EndpointConfig(
        id="meditron",
        model="epfl-llm/meditron-7b",
        url=os.getenv("LOCAL_VLLM_JUDGE3_URL", "http://localhost:8003"),
        description="Meditron 7B -- EPFL medical LLM (chen2023meditron70b)",
    ),
    EndpointConfig(
        id="medalpaca",
        model="medalpaca/medalpaca-7b",
        url=os.getenv("LOCAL_VLLM_JUDGE4_URL", "http://localhost:8004"),
        description="MedAlpaca 7B -- medical instruction-tuned LLaMA (han2023medalpaca)",
    ),
]


def get_judge_configs_as_dicts():
    """Return judge list as plain dicts for JSON config files."""
    return [
        {"id": e.id, "model": e.model, "url": e.url,
         "max_tokens": e.max_tokens, "temperature": e.temperature,
         "timeout_seconds": e.timeout_seconds}
        for e in DEFAULT_JUDGE_ENDPOINTS
    ]
