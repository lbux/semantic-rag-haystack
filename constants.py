from haystack_integrations.components.evaluators.ragas import RagasMetric
from haystack_integrations.components.evaluators.uptrain import UpTrainMetric

DEFAULT_UPTRAIN_METRIC = UpTrainMetric.FACTUAL_ACCURACY

DEFAULT_RAGAS_METRIC = RagasMetric.FAITHFULNESS

MODEL_NAME = "WizardLM-2-7B.Q8_0.gguf"

MODEL_GGUF_URL = "https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF/resolve/main/WizardLM-2-7B.Q8_0.gguf"
