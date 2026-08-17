from collections import defaultdict
import tiktoken
from pathlib import Path
import json
import random

ic = r"D:\Tesi PY\data\Cluster_Da_Pullire.json"
IC = Path(ic)
im = r"D:\Tesi PY\Ground_truth\C_GROUND_TRUTH_F.json"
IM = Path(im)
NER_type = False
with_context = False
_encoder = tiktoken.get_encoding("cl100k_base")

promptC = """Devi verificare che le menzioni dei cluster si riferiscano correttamente al cluster in cui si trovano.
In output crea un dict(str, list), la lista di interi, e inserisci:
in ELIMINARE le menzioni il cui text NON è presente nel proprio context, se c'è. 
in SPOSTARE le menzioni il cui text è giusto ma NON si rifersice al cluster in cui si trova.
OUTPUT es:
{"ELIMINARE":[ID_ES1, ID_ES2], "SPOSTARE": [ID_ES3, ID_ES4]}
Rispondi SOLO CON JSON VALIDO, senza testo prima o dopo, senza esempi, senza markdown.
Input:
{}"""

promptM = """In Input hai una lista di titoli di Cluster, associati al loro id
es: [[id_es1, "Presidente del Consiglio"],[id_es2,"Marco Ramat"],[id_es3,"Francesco Cossiga"], [id_es4, "Primo Ministro"]]
crea un list[list] in cui ogn lista contiene gli ID dei cluster che riferiscono alla stessa entità.
Nel dict di output ci devono essere tutti gli elementi dati in input, se un cluster è l'unico a rifereisisi ad un'entità allora mettilo singolo
esempio Output:
[[ides_1, ides_45], [ides_4, ides_7, ides_89], [ides_76], [ides_21]]
Restituisci ESCLUSIVAMENTE LA LISTA DI LISTE, nessun testo prima o dopo, nessun esempio, nessun markdown.
Input:
{}"""

def run():
    with IC.open(encoding="utf-8") as f:
        clusters: list = json.load(f)
    total_tokens = 0
    mentions: dict[int, list] = defaultdict(list)
    n = 0
    for cluster in clusters:
        del cluster["originalDocId"]
        del cluster["clusterId"]
        if not NER_type:
            del cluster["type"]
        for mention in cluster.get("mentions"):
            del mention["start"]
            del mention["end"]
            del mention["url"]
            if not with_context:
                del mention["context"]
        if  len(mentions[n]) >= 10:
            n += 1
        mentions[n].append(cluster)
    for _, batch in mentions.items():
        PromptC = promptC.replace("{}", json.dumps(batch, ensure_ascii=False, indent=2), 1)
        total_tokens += count_tokens(PromptC)
    with IM.open(encoding="utf-8") as f:
        gt: dict = json.load(f)
    sep_d = separate_doc(gt)
    sep_c : dict[str, dict[int, list[tuple[str, str]]]] = {doc_id: separate_clusters(clusters) for doc_id, clusters in sep_d.items()}
    for doc_id, clusters_lists in sep_c.items():
        workers = len(clusters_lists.keys())
        n = 0
        while True:
            total_tokens += count_tokens(promptM.replace("{}", json.dumps(clusters_lists[n], ensure_ascii=False, indent=2), 1))
            n += 1
            if workers == 1:
                break
            workers = workers // 2
            clusters_lists = separate_clusters(sep_d[doc_id], P = workers)

    sep_a: dict[int, list[tuple[str, str]]] = separate_clusters(gt)
    workers = len(sep_a.keys())
    for _, clusters_list in sep_a.items():
        total_tokens += count_tokens(promptM.replace("{}", json.dumps(clusters_list, ensure_ascii=False, indent=2), 1))
    while True:
        if workers == 1:
            break
        workers = workers // 2
        sep_a = separate_clusters(gt, P = workers)
        for _, clusters_list in sep_a.items():
            total_tokens += count_tokens(promptM.replace("{}", json.dumps(clusters_list, ensure_ascii=False, indent=2), 1))

    print(total_tokens)
    
def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento

    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    return doc_clusters

def separate_clusters(clusters: list, batch_length: int = 50, P: int = 0) -> dict: # separa i singoli cluster

    none = 0
    titles: dict[int, list[tuple[int, str]]] = defaultdict(list)
    if P == 0:
        n = 0
        for cluster in clusters:
            if cluster is None:
                none += 1
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            id_sliced = id[-15:]
            if len(titles[n]) >= batch_length:
                titles[n + 1].append((id_sliced, title))
                random.shuffle(titles[n + 1])
                n += 1
            else:
                titles[n].append((id_sliced, title))
                random.shuffle(titles[n])
    else:
        P -= 1
        n = 0
        for cluster in clusters:
            if cluster is None:
                none += 1
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            id_sliced = id[-15:]
            if len(titles[n]) >= 50:
                if n == P:
                    n = -1
                titles[n + 1].append((id_sliced, title))
                random.shuffle(titles[n + 1])
                n += 1
            else:
                titles[n].append((id_sliced, title))
                random.shuffle(titles[n])
    return titles

def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


if __name__ == "__main__" :
    run()
