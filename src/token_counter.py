from collections import defaultdict
from pathlib import Path
import json

c = 0
m = 0
with Path(r"D:\Tesi PY\Data_completo\Clusters_EPSTEIN.json").open(encoding="utf-8") as f:
    clusters: list = json.load(f)
for cluster in clusters:
    if cluster is not None:
        c += 1
        for mention in cluster.get("mentions"):
            if mention is not None:
                m += 1
print(c, m)
Ic = 107.7 * m + 22881
Oc = 0.443 * m + 635
Im = 547.1 * c - 182118
Om = 209.7 * c - 63095
print("Cleaning: I token : ", str(Ic), " | O token : ", str(Oc))
print("Merging: I token : ", str(Im), " | O token : ", str(Om))
costo_clean = ((109.2 * m + 13245) * 0.319) / 1000000
costo_merge = ((739.1 * c - 191876) * 2.494) / 1000000
print("Costo cleaning: ", str(costo_clean), " | Costo merging: ", str(costo_merge))