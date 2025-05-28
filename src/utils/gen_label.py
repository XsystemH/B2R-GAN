import json

labels = {"images": []}

for i in range(2):
  for j in range(7):
    for k in range(5):
      id = i*7*5 + j*5 + k
      entry = {}
      entry["id"] = id
      entry["x"] = j * 50.0 + 20.0 * (i == 1)
      entry["y"] = k * 50.0
      entry["theta"] = 45.0 * (i == 1)
      labels["images"].append(entry)

outpath = "data/train/070502/labels.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)
    print(f"Labels saved to {outpath}")