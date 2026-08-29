import json
from pathlib import Path

count = 0
count1 = 0
with Path(r"D:\Tesi PY\data\Cluster_Da_Pullire.json").open(encoding="utf-8") as f:
    clusters: list = json.load(f)
for cluster in clusters:
    if cluster is not None:
        count1 += 1
        for mention in cluster.get("mentions"):
            if mention is not None:
                count += 1
print(count1, count)