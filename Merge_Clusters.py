from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
from pathlib import Path
from collections import defaultdict
from google import genai
import anthropic

OUTPUT_PATH = r"D:\Tesi PY\Clusters_Uniti.json"
API_KEY = "genai_API_key"
MODELLO0 = "gemini-3.5-flash"
MODELLO1 =  "gemini-3.1-pro-preview"
MODELLO2 = "gemini-3.1-flash-lite"
MODELLO3 =  "gemini-3-flash-preview"
MODELS = [MODELLO0, MODELLO1, MODELLO2, MODELLO3]
CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"]  # dal più economico al più capace

prompt = """In Input hai una lista di  titoli di Cluster, di uno stesso tipo, associati al loro id
es: "persona": [[id_es1, "Presidente del Consiglio"],[id_es2,"Marco Ramat"],[id_es3,"Francesco Cossiga"], [id_es4, "Primo Ministro"]]
crea un dict[str, list] in cui c'è come chiave un l'id del titolo principale e nella lista tutti gli id
dei titoli dei cluster che riferiscono alla stessa entità del principale.
Nel dict di output ci devono essere tutti gli elementi dati in input, se un cluster è l'unico a rifereisisi ad un'entità allora metti es: (id: [])
esempio Output:
(id1: [id3, id4, id6], id4: [id57, id89], id7: [])
Un titolo principale non può far parte della lista di un altro titolo prinicpale, e un titolo presente nella lista di un'altro titolo non può essere presente come titolo principale.
es( id1: [id2], id2:[] è erratto id2:[] non deve essere presente)
Restituisci ESCLUSIVAMENTE il JSON, nessun testo prima o dopo, nessun esempio, nessun markdown.
Input:
{}"""

volta_lock = threading.Lock() #DEBUG
volta = 0  #DEBUG
model_curr = 0
client = genai.Client(api_key=API_KEY)
client_claude = anthropic.Anthropic(api_key="anthropic_api_key")

def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento

    print("inizio separate_doc")

    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    print("fine separate_doc")

    return doc_clusters

def separate_for_types(clusters: list, version: int = 0) -> dict: # separa le singole menzioni

    if version == 0:
        print("inizio separate_for_types")
        titles: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for cluster in clusters:
            st = cluster.get("type")
            title = cluster.get("title")
            id = cluster.get("clusterId")
            id_sliced = id[len(id)-15:]
            titles[st].append((id_sliced, title))

        print("fine separate_for_types")

    else:
        titles: dict[str, list] = defaultdict(list)
        for cluster in clusters:
            st = cluster.get("type")
            titles[st].append(cluster)

    return titles

def counter(to_merge: dict[str, dict[str, list]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for type, _ in to_merge.items():
        c[type[:type.index("_")]] += 1
    return c

def process_clusters_parallel(Sep_C: dict[str, list[tuple[str, str]]], max_workers: int, version: int = 0) -> dict:

    if version == 0:
        print("inizio process_cluster_parallel")

        to_merge: dict[str, list] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = {}
            for type, cluster_list in Sep_C.items():
                future_obj = executor.submit(call_llm, cluster_list, prompt)
                tasks[future_obj] = (type)

            for task in as_completed(tasks):
                try:
                    result = task.result()
                    for key, value_list in result.items():
                        to_merge[key].extend(value_list)
                except Exception as e:
                    print("Errore nella chiamata al modello per il tipo:", tasks[task], "-", e)
        print("fine process_cluster_parallel")

    else:
        to_merge: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for type, cluster_list in Sep_C.items():
            if len(cluster_list) / 200 > 1.5:
                print("CLSUTER TROPPO GRANDE, DIVISIONE IN PEZZI PIù PICCOLI")
                n = 0
                for i in range(0, len(cluster_list), 200):
                    other_list = cluster_list[i:i+200]
                    Sep_C[type + "*" + str(n)] = other_list
                    n += 1
                    max_workers += 1
                del Sep_C[type]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = {}
            for type, cluster_list in Sep_C.items():
                future_obj = executor.submit(call_llm, cluster_list, prompt)
                tasks[future_obj] = (type)

            for task in as_completed(tasks):
                try:
                    result = task.result()
                    S_type = tasks[task]
                    for key, value_list in result.items():
                        to_merge[S_type][key].extend(value_list)
                except Exception as e:
                    print("Errore nella chiamata al modello per il tipo:", tasks[task], "-", e)

        print("fine process_cluster_parallel")

    return to_merge

def call_llm(cluster_list: list, prompt: str, retry: int = 10, delay: float = 5.0) -> dict:
    
    global volta
    with volta_lock:
        volta += 1
        n = volta

    Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
    print("chiamata modello per menzioni n. " + str(n))

    for model in CLAUDE_MODELS:
        for attempt in range(retry):
            try:
                response = client_claude.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": Prompt}]
                )
                text = response.content[0].text.strip()
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("Risposta senza JSON valido.")
                Result = json.loads(text)
                print(f"fine chiamata n. {n}")
                return Result

            except Exception as e:
                err = str(e)
                if "529" in err or "overloaded" in err.lower():
                    print(model + " sovraccarico")
                    break
                if "rate" in err.lower() or "429" in err:
                    print(model + " rate limit")
                    break
                if attempt < retry-1:
                    time.sleep(delay)
                else:
                    print(f"Errore persistente con {model}: {e}")
                    break

    raise RuntimeError("Tutti i modelli hanno esaurito la quota.")

def merge_clusters(doc_clusters: list, to_merge: dict[str, list], version : int = 0) -> list:

    def find_cluster(id: str):
        for c in doc_clusters:
            x = c.get("clusterId")
            y = x[len(x)-15:]
            if y == id:
                return c
        return None

    N_clusters = []
    absorbed_ids = set()
    principal_ids = set()

    for principal_id, clusters_ids in to_merge.items():
        P_cluster = find_cluster(principal_id)
        if P_cluster is None:
            continue
        for sub_id in clusters_ids:
            S_cluster = find_cluster(sub_id)
            if S_cluster is None:
                continue
            P_cluster["mentions"].extend(S_cluster["mentions"])
            principal_ids.add(principal_id)
            if version == 1:
                print("ENTRATO VERSION 1 MERGE CLUSTERS")
                P_cluster["originalDocId"] = "Extra_Document_Cluster"
                P_cluster["clusterId"] = "bo"
            absorbed_ids.add(sub_id)
        N_clusters.append(P_cluster)

    for c in doc_clusters:
        x = c.get("clusterId")
        y = x[len(x)-15:]
        if y not in absorbed_ids and y not in principal_ids:

            N_clusters.append(c)

    return N_clusters

def run(input_path: str, output_path: str = OUTPUT_PATH):
    inizio = time.perf_counter()

    I = Path(input_path)
    O = Path(output_path)

    with I.open(encoding="utf-8") as f:
        clusters: list = json.load(f)

    Sep_D: dict[str, list] = separate_doc(clusters)

    Sep_C: dict[str, dict[str, list[tuple[str, str]]]] = {doc_id: separate_for_types(clusters) for doc_id, clusters in Sep_D.items()}
        # dict[doc_id, dict[type, list[tuple[id, clutser_title]]]]

    Merged_documents: list = []

    for doc_id, clusters_lists in Sep_C.items():
        print("incomicio lavoro su doc:", doc_id)
        workers = len(clusters_lists)
        to_merge: dict[str, list] = process_clusters_parallel(clusters_lists, workers)
        document_clusters = Sep_D[doc_id]
        merged_clusters : list = merge_clusters(document_clusters, to_merge)

        Merged_documents.extend(merged_clusters)

    Sep_A_C: dict[str, list[tuple[str, str]]] = separate_for_types(Merged_documents)
           # dict[type, list[tuple[id, clutser_title]]]
    Types_Clusters: dict[str, list] = separate_for_types(Merged_documents, version = 1)
                 # dict[type, clusters]
    workers = len(Sep_A_C)
    to_merge: dict[str, dict[str, list]] = process_clusters_parallel(Sep_A_C, workers, version = 1)
            # dict[type, dict[pirincipal_id, sub_ids]]
    Merged_A_Documents: list = []
    for type, clusters in Types_Clusters.items():
        print("inizio merge per tipo:", type)
        try:
            if to_merge.get(type[:type.index("*")]):
                type_n = type[:type.index("*")]
                merged_A_clusters: list = merge_clusters(clusters, to_merge.get(type_n), version = 1)
        except:
            merged_A_clusters: list = merge_clusters(clusters, to_merge.get(type), version = 1)
            del to_merge[type]
        
        Merged_A_Documents.extend(merged_A_clusters)

    final_clusters: dict[str, list] = defaultdict(list)
    if to_merge != {}:
        print(to_merge)
    while to_merge != {}:
        Sep_T_C : dict[str, list[tuple[str, str]]] = defaultdict(list)
        n = 0
        counter_result: dict[str, int] = counter(to_merge)
        T_A_clusters : dict[str, list] = defaultdict(list)
        for type, ids in to_merge.items():
            T_clusters: list = Types_Clusters.get(type[:type.index("*")])
            T_A_clusters[type[:type.index("*")]].extend(T_clusters)
            for id_key in ids.keys():
                for c in T_clusters:
                    x = c.get("clusterId")
                    if x[len(x)-15:] == id_key:
                        if counter_result[type[:type.index("*")]] == 1:
                            Sep_T_C[type[:type.index("*")] + "*" + "final"].append((id_key, c.get("title")))
                        elif int(type[type.index("*")+1:]) % 2 == 0:
                            Sep_T_C[type[:type.index("*")] + "*" + str(n)].append((id_key, c.get("title")))
                        else:
                            Sep_T_C[type[:type.index("*")] + "*" + str(n)].append((id_key, c.get("title")))
                            n += 1
            
        workers = len(Sep_T_C)
        to_merge: dict[str, dict[str, list]] = process_clusters_parallel(Sep_T_C, workers, version = 1)
        for type, clusters in T_A_clusters.items():
            try:
                if to_merge.get(type[:type.index("*final")]):
                    type_n = type[:type.index("*")]
                    merged_A_clusters: list = merge_clusters(clusters, to_merge.get(type_n), version = 1)
                    del to_merge[type]
            except:
                try:
                    if to_merge.get(type[:type.index("*")]):
                        type_n = type[:type.index("*")]
                        merged_A_clusters: list = merge_clusters(clusters, to_merge.get(type_n), version = 1)
                except:
                    merged_A_clusters: list = merge_clusters(clusters, to_merge.get(type), version = 1)
                    del to_merge[type]

            final_clusters[type[:type.index("*")]] = merged_A_clusters   
    for type, clusters in final_clusters.items():
        Merged_A_Documents.extend(clusters)

    with O.open("w", encoding="utf-8") as f:
        json.dump(Merged_A_Documents, f, ensure_ascii=False, indent=2)
    
    fine = time.perf_counter()
    print(f"Tempo di MERGE_CLUSTERS: {(fine - inizio)/60:.2f} minuti")


if __name__ == "__main__" :
    run(r"D:\Tesi PY\Cluster_puliti_manualmente.json")
