
import json

N_SAMPLES = 6000


with open("data/train.computed.json", "r") as f:
    train_data = json.load(f)
    
    
subset = train_data[:N_SAMPLES]



with open(f"data/train.computed.{N_SAMPLES}.json", "w") as f:
    json.dump(subset, f)    