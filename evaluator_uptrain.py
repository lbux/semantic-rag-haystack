# Abandoned code for evaluating the generated data using the uptrain library

import json
import re

from uptrain import EvalLLM, Evals, RcaTemplate, ResponseMatching, Settings

from utils import read_serialized_generated_answer

queries, documents, answers, staff_answers = read_serialized_generated_answer(
    "generated_data_2024-04-25_21-26-25.json"
)

data = []

settings = Settings(model="ollama/llama3")
eval_llm = EvalLLM(settings=settings)

results = eval_llm.evaluate(data=data, checks=[Evals.CONTEXT_RELEVANCE])

with open("documents\evaluation_output\output.json", "w") as f:
    f.write(json.dumps(results, indent=4))
