import os
import json
from tqdm import tqdm


VAE_DIR = "/mnt/disks/training-data-refine/RAPCHAT_DATASET_VAE"

with open(f"data/train.json", 'r') as f:
    train_data = json.load(f)
    
    
print(train_data[0])

print(os.path.basename(train_data[0]["location"]))


counter = {
    False: 0,
    True: 0
}

computed_files = []
uncomputed_files = []



for item in tqdm(train_data):
    base_name = os.path.basename(item["location"])
    vae_file = os.path.join(
        VAE_DIR, base_name.replace(".wav", "_latent.npz")
    )
    
    if os.path.exists(vae_file):
        computed_files.append(item)
    else:
        uncomputed_files.append(item)
        


with open("train.computed.json", "w") as f:
    json.dump(computed_files, f)
    
    
with open("train.uncomputed.json", "w") as f:
    json.dump(uncomputed_files, f)
    




