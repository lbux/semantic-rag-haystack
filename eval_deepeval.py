import json

import llama_cpp
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from utils import read_serialized_generated_answer

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


class MetaLLamaInstruct(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model = llama_cpp.Llama(
            model_path=model_path,
            use_mmap=True,
            verbose=True,
            chat_format="llama-3",
            n_ctx=8000,
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


query, documents, answers, staff_answers = read_serialized_generated_answer(
    "generated_data_2024-04-25_21-26-25.json"
)

llama3 = MetaLLamaInstruct(model_path="models/Meta-Llama-3-8B-Instruct.Q8_0.gguf")

contextual_precision = ContextualPrecisionMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)
contextual_recall = ContextualRecallMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)
contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)

test_cases = []
for i in range(len(query)):
    test_case = LLMTestCase(
        input=query[i],
        actual_output=answers[i],
        expected_output=staff_answers[i],
        retrieval_context=documents[i],
    )
    test_cases.append(test_case)

# Evaluate all test cases
results = evaluate(
    test_cases=test_cases,
    metrics=[contextual_precision, contextual_recall, contextual_relevancy],
    ignore_errors=True,
    print_results=True,
)
