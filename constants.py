from haystack_integrations.components.evaluators.uptrain import UpTrainMetric
from haystack_integrations.components.evaluators.ragas import RagasMetric

DEFAULT_UPTRAIN_METRIC = UpTrainMetric.FACTUAL_ACCURACY

DEFAULT_RAGAS_METRIC = RagasMetric.FAITHFULNESS

MODEL_NAME = "mistral-7b-instruct-v0.2.Q8_0.gguf"

MODEL_GGUF_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf"