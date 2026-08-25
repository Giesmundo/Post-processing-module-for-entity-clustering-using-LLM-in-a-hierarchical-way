import json
from pathlib import Path


I = Path(r"D:\Tesi PY\doc14.json")
with I.open(encoding="utf-8") as f:
    clusters : list = json.load(f)
GTO = Path(r"D:\Tesi PY\Ground_truth\C_GROUND_TRUTH_O.json")
with GTO.open(encoding="utf-8") as f:
    gto : dict[str, dict[str, str]] = json.load(f)
GTF = Path(r"D:\Tesi PY\Ground_truth\C_GROUND_TRUTH_F.json")
with GTF.open(encoding="utf-8") as f:
    gtf = json.load(f)

# PARTE DI PULIZIA DELLE MENZIONI
X = 0
new_clusters : list = []
for cluster in clusters:
    if cluster is not None:
        t = cluster.get("title")
        kept_mentions = []
        for mention in cluster.get("mentions"):
            if mention is not None:
                m = mention.get("text")
                c = mention.get("context")
                print("-@- TITOLO : ", f"\033[32m {t} \033[0m", " MENZIONE : ", f"\033[32m {m} \033[0m", " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("CONTESTO : ", c)
                while True:
                    try:
                        b = input("va bene? ------ ")
                        while b not in ("y", "n", "e"): 
                            b = input("o si o no")
                        break
                    except KeyboardInterrupt:
                        continue
                if b == "y":
                    kept_mentions.append(mention)
                elif b == "n":
                    new_cluster = {
                        "originalDocId": cluster.get("originalDocId"),
                        "clusterId" : cluster.get("clusterId") + "_orphan_" + str(X),
                        "title" : m,
                        "type" : cluster.get("type"),
                        "mentions" : [mention]
                    }
                    gto.setdefault(cluster.get("originalDocId"), {})[str(mention.get("id"))] = "SPOSTARE"
                    new_clusters.append(new_cluster)
                    X += 1
                else:
                    gto.setdefault(cluster.get("originalDocId"), {})[str(mention.get("id"))] = "ELIMINARE"
            cluster["mentions"] = kept_mentions
    
clusters.extend(new_clusters)
gtf.extend(clusters)

with GTF.open("w", encoding="utf-8") as f:
    json.dump(gtf, f, ensure_ascii=False, indent=2)
with GTO.open("w", encoding="utf-8") as f:
    json.dump(gto, f, ensure_ascii=False, indent=2)

# PRENDERE I TITOLI
GTM = Path(r"D:\Tesi PY\Ground_truth\M_GROUND_TRUTH.json")

with GTM.open(encoding="utf-8") as f:
    gtm : dict[str, list] = json.load(f)
with Path(r"D:\Tesi PY\titoli.txt").open("w", encoding="utf-8") as f:
    f.write(f"\n")

for cluster in clusters:
    title = cluster.get("title")
    print(f"\n TITOLO:\033[32m {title} \033[0m")
    while True:
        try:
            annotation = input("Chiave GT da associare (invio per nessuna): ")
            break
        except KeyboardInterrupt:
            continue
    if not annotation:
        with Path(r"D:\Tesi PY\titoli.txt").open("a", encoding="utf-8") as f:
            f.write(f"{title}\n")        
        gtm.setdefault(title, []).extend(cluster.get("mentions"))
    else:
        try:
            gtm[annotation].extend(cluster.get("mentions"))
        except:
            while True:
                print(annotation, " non è valida")
                annotation = input("Inserisci una chiave valida: ")
                try:
                    gtm[annotation].extend(cluster.get("mentions"))
                    break
                except:
                    continue

bis_clusters : dict[str, list] = {}
for _, mentions in gtm.items():
    if not mentions:
        continue
    longest_mention = max(mentions, key=lambda m: len(m.get("text")))
    new_title = longest_mention.get("text")
    bis_clusters[new_title] = mentions

with GTM.open("w", encoding="utf-8") as f:
    json.dump(bis_clusters, f, ensure_ascii=False, indent=2)

with I.open("w", encoding="utf-8") as f:
    json.dump(clusters, f, ensure_ascii=False, indent=2)