from haystack_integrations.components.evaluators.ragas import RagasMetric
from haystack_integrations.components.evaluators.uptrain import UpTrainMetric

DEFAULT_UPTRAIN_METRIC = UpTrainMetric.FACTUAL_ACCURACY

DEFAULT_RAGAS_METRIC = RagasMetric.FAITHFULNESS

MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"

MODEL_GGUF_URL = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
