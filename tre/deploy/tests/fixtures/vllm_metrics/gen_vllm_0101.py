"""Regenerate vllm-0.10.1.prom without a GPU (not a test; pytest does not collect it).

    docker run --rm -v "$PWD":/w --entrypoint python3 vllm/vllm-openai:0.10.1 \
        /w/gen_vllm_0101.py > body.prom

then drop any vLLM log line from the output and prepend the fixture header comment.
"""
import sys
from unittest.mock import MagicMock

import prometheus_client
from vllm.v1.metrics.loggers import PrometheusStatLogger

cfg = MagicMock()
cfg.observability_config.show_hidden_metrics = False
cfg.model_config.served_model_name = "dsqwen-7b"
cfg.model_config.max_model_len = 16384
cfg.speculative_config = None
cfg.lora_config = None
cfg.cache_config.num_gpu_blocks = 1000
PrometheusStatLogger(cfg, [0])
sys.stdout.write(prometheus_client.generate_latest().decode())
