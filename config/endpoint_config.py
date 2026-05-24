# vLLM endpoint config for the four judge models
# each judge runs on its own port via a separate vLLM server
# see run_all.sh for how to launch them

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    id: str
    model: str
    url: str
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    description: Optional[str] = None


# judge panel from the paper (Sec 4)
# four locally deployable SLMs - no data leaves the machine
DEFAULT_JUDGE_ENDPOINTS = [
    EndpointConfig(
        id='medgemma',
        model='google/medgemma-4b-it',
        url=os.getenv('LOCAL_VLLM_JUDGE1_URL', 'http://localhost:8001'),
        description='MedGemma 4B (sellergren2025medgemma)',
    ),
    EndpointConfig(
        id='biomistral',
        model='BioMistral/BioMistral-7B-DARE',
        url=os.getenv('LOCAL_VLLM_JUDGE2_URL', 'http://localhost:8002'),
        description='BioMistral 7B-DARE (labrak2024biomistral)',
    ),
    EndpointConfig(
        id='meditron',
        model='epfl-llm/meditron-7b',
        url=os.getenv('LOCAL_VLLM_JUDGE3_URL', 'http://localhost:8003'),
        description='Meditron 7B (chen2023meditron70b)',
    ),
    EndpointConfig(
        id='medalpaca',
        model='medalpaca/medalpaca-7b',
        url=os.getenv('LOCAL_VLLM_JUDGE4_URL', 'http://localhost:8004'),
        description='MedAlpaca 7B (han2023medalpaca)',
    ),
]


def get_judge_configs_as_dicts():
    return [
        {
            'id': e.id, 'model': e.model, 'url': e.url,
            'max_tokens': e.max_tokens, 'temperature': e.temperature,
            'timeout_seconds': e.timeout_seconds,
        }
        for e in DEFAULT_JUDGE_ENDPOINTS
    ]
