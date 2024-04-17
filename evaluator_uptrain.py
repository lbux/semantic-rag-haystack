from uptrain import EvalLLM, Evals, Settings
import json
from utils import read_serialized_generated_answer, serialize_evaluation_results

queries, documents, answers, staff_answers = read_serialized_generated_answer("")

settings = Settings(model='ollama/...')
eval_llm = EvalLLM(settings=settings)

results = eval_llm.evaluate(project_name = "uptrain local test", data=data, checks=[Evals.FACTUAL_ACCURACY])

print(results)