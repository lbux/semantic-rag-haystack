import os
from haystack import Pipeline

from utils import read_serialized_data

from getpass import getpass

from haystack_integrations.components.evaluators.uptrain import (
    UpTrainEvaluator,
    UpTrainMetric,
)

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

evaluator = UpTrainEvaluator(
    metric=UpTrainMetric.FACTUAL_ACCURACY,
    api="openai",
)

evaluator_pipeline = Pipeline()
evaluator_pipeline.add_component("evaluator", evaluator)

queries, documents, answers = read_serialized_data()

evaluation_results = evaluator_pipeline.run(
    {
        "evaluator": {
            "questions": queries,
            "contexts": documents,
            "responses": answers,
        }
    }
)

print(evaluation_results)
