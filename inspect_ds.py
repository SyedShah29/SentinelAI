from datasets import load_dataset

print("--- RAID Dataset ---")
ds_raid = load_dataset("liamdugan/raid", split="train")
models_raid = set(ds_raid["model"])
print(f"Total samples: {len(ds_raid)}")
print("Models in RAID:", models_raid)

print("\n--- AI Model Accent Corpus ---")
ds_accent = load_dataset("firatmihci/ai-model-accent-corpus", split="train")
models_accent = set(ds_accent["model"])
print(f"Total samples: {len(ds_accent)}")
print("Models in Accent:", models_accent)
