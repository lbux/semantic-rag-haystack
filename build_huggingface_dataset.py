from datasets import load_dataset  
from utils import (read_input_json)

data = read_input_json("summer21.json")
print(data)

dataset = load_dataset("json", data_files="summer21.json", split="train")
print(dataset)