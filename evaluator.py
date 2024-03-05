import os
from haystack import Pipeline
from constants import *

from utils import (
    read_serialized_generated_answer,
    metric_to_params,
    serialize_evaluation_results,
)

from getpass import getpass

from haystack_integrations.components.evaluators.uptrain import UpTrainEvaluator

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

evaluator = UpTrainEvaluator(
    metric=DEFAULT_METRIC,
    api="openai",
)

evaluator_pipeline = Pipeline()
evaluator_pipeline.add_component("evaluator", evaluator)

queries, documents, answers = read_serialized_generated_answer()

ground_truths = []

evaluation_paramaters = metric_to_params(
    DEFAULT_METRIC,
    {
        "questions": queries,
        "contexts": documents,
        "responses": answers,
        "ground_truths": ground_truths,
    },
)

evaluator_pipeline.draw("evaluator_pipeline.png")

evaluation_results = evaluator_pipeline.run(evaluation_paramaters)

serialize_evaluation_results(evaluation_results)
