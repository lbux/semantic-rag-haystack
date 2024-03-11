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
    api="openai"
)

evaluator_pipeline = Pipeline()
evaluator_pipeline.add_component("evaluator", evaluator)

queries, documents, answers, staff_answers = read_serialized_generated_answer()


evaluation_paramaters = metric_to_params(
    DEFAULT_METRIC,
    {
        "questions": queries,
        "contexts": documents,
        "responses": answers,
        "ground_truths": staff_answers,
    },
)

evaluator_pipeline.draw("evaluator_pipeline.png")

evaluation_results = evaluator_pipeline.run(evaluation_paramaters)

for i, group in enumerate(evaluation_results['evaluator']['results']):
    for result in group:
        result['original_question'] = queries[i]
        result['llm_response'] = answers[i]
        result['staff_answer'] = staff_answers[i] if staff_answers[i] else "No staff answer provided"


serialize_evaluation_results(evaluation_results)
