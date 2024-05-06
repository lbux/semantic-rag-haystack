import argparse
import os

import llama_cpp
from deepeval.models.base_model import DeepEvalBaseLLM
from haystack import Pipeline
from haystack_integrations.components.evaluators.deepeval import (
    DeepEvalEvaluator,
    DeepEvalMetric,
)

from utils import read_serialized_generated_answer, serialize_evaluation_results

json_schema = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["yes", "no"]},
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


class LlamaCppModel(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model = llama_cpp.Llama(
            model_path=model_path,
            use_mmap=True,
            verbose=True,
            chat_format="llama-3",
            n_ctx=8092,
            n_gpu_layers=-1,
        )
        super().__init__()

    def get_model_name(self):
        return "Meta-Llama-3-8B-Instruct"

    def load_model(self):
        return self.model

    def generate(self, prompt):
        response = self.model.create_chat_completion(
            messages=[{"role": "system", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
        )
        print(response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]

    async def a_generate(self, prompt):
        return self.generate(prompt)


llama3 = LlamaCppModel(model_path="models/Meta-Llama-3-8B-Instruct-Q8_0.gguf")

queries, documents, answers, staff_answers = read_serialized_generated_answer(
    "generated_data_2024-05-05_17-27-16.json"
)

eval_pipeline = Pipeline()

evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.CONTEXTUAL_PRECISION, metric_params={"model": llama3}
)

eval_pipeline.add_component("evaluator", evaluator)

results = eval_pipeline.run(
    {
        "evaluator": {
            "questions": queries,
            "contexts": documents,
            "responses": answers,
            "ground_truths": staff_answers,
        }
    }
)

for i, group in enumerate(results["evaluator"]["results"]):
    for result in group:
        result["original_question"] = queries[i]
        result["llm_response"] = answers[i]
        result["staff_answer"] = (
            staff_answers[i] if staff_answers[i] else "No staff answer provided"
        )

serialize_evaluation_results(results)
