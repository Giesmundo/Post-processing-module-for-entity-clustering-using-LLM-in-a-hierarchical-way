from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
from pathlib import Path
from collections import defaultdict
from google import genai
import anthropic
import random

OUTPUT_PATH = r"D:\Tesi PY\Clusters_Uniti.json"
GT = r"D:\Tesi PY\M_GROUND_TRUTH.json"
MODELS = ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]
CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"]

prompt = """In Input hai una lista di titoli di Cluster, associati al loro id
es: [[id_es1, "Presidente del Consiglio"],[id_es2,"Marco Ramat"],[id_es3,"Francesco Cossiga"], [id_es4, "Primo Ministro"]]
crea un list[list] in cui ogn lista contiene gli ID dei cluster che riferiscono alla stessa entità.
Nel dict di output ci devono essere tutti gli elementi dati in input, se un cluster è l'unico a rifereisisi ad un'entità allora mettilo singolo
esempio Output:
[[ides_1, ides_45], [ides_4, ides_7, ides_89], [ides_76], [ides_21]]
Restituisci ESCLUSIVAMENTE LA LISTA DI LISTE, nessun testo prima o dopo, nessun esempio, nessun markdown.
Input:
{}"""

volta_lock = threading.Lock() #DEBUG
volta = 0  #DEBUG
m = 0
modello_corrente= 0
client = genai.Client(api_key="genai_api_key")
client_claude = anthropic.Anthropic(api_key="anthropic_api_key")

def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento

    print("inizio separate_doc")

    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    print("fine separate_doc")

    return doc_clusters

def separate_clusters(clusters: list, P: int = 0) -> dict: # separa i singoli cluster

    none = 0
    titles: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if P == 0:
        n = 0
        for cluster in clusters:
            if cluster is None:
                none += 1
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            id_sliced = id[-15:]
            if len(titles[n]) >= 50:
                titles[n + 1].append((id_sliced, title))
                random.shuffle(titles[n + 1])
                n += 1
            else:
                titles[n].append((id_sliced, title))
                random.shuffle(titles[n])
    else:
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
    print("cluster nulli ", none)
    return titles

def process_clusters_parallel(Sep_C: dict[int, list[tuple[str, str]]], max_workers: int) -> list[list]:

    print("inizio process_cluster_parallel")

    to_merge: list[list] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {}
        for batch_number, cluster_list in Sep_C.items():
            future_obj = executor.submit(A_call_llm, cluster_list, prompt)
            tasks[future_obj] = (batch_number)

        for task in as_completed(tasks):
            try:
                result = task.result()
                to_merge.extend(result)
            except Exception as e:
                print("Errore nella chiamata al modello nel batch: ", tasks[task], " - ", e)
    print("fine process_cluster_parallel")

    return to_merge

def A_call_llm(cluster_list: list, prompt: str, retry: int = 10, delay: float = 5.0) -> dict:
    
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
                start = text.find('[')
                end = text.rfind(']')
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

def G_call_llm(cluster_list: list, prompt: str, retry: int = 5, delay: float = 5.0) -> str:
    global modello_corrente
    global volta
    with volta_lock:
        volta += 1
        n = volta
    print("chiamata numero: ", str(n))

    for i in range(modello_corrente, len(MODELS)):
        modello = MODELS[i]
        for attempt in range(retry):
            try:
                Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
                response = client.models.generate_content(model=modello, contents=Prompt)
                text = response.text.strip()
                #clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("La risposta del modello non contiene un JSON valido.")
                Result = json.loads(text)
                return Result
            
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(modello + " esaurito")
                    modello_corrente += 1
                    break
                print(f"[{modello}] Tentativo {attempt+1}/{retry} fallito: {e}")
                if attempt < retry - 1:
                    time.sleep(delay)
                else:
                    modello_corrente += 1
                    break
    raise RuntimeError("Tutti i modelli hanno esaurito la quota.")

def merge_clusters(doc_clusters: list, to_merge: list[list], version : int = 0, doc_id : str = "",) -> list:
    p = 0
    global m

    def find_cluster(id: str):
        for c in doc_clusters:
            if c is None:
                continue
            x = c.get("clusterId")
            if x is None:
                continue
            if x[-15:] == id:
                return c

    N_clusters = []
    O_clusters = []

    for ids in to_merge:
        if len(ids) == 1:
            c = find_cluster(ids[0])
            O_clusters.append(c)
            continue

        to_merge_set = []
        mentions: list = []
        for cid in ids:
            cluster = find_cluster(cid)
            if cluster is not None:
                to_merge_set.append(cluster)
            else:
                print("cluster non trovato ", cid)
                p += 1

        if not to_merge_set:
            continue

        title = ""
        for c in to_merge_set:
            mentions.extend(c.get("mentions"))
            t = c.get("title")
            if len(t) > len(title):
                title = t

        if version == 0:
            new_cluster = {
                "originalDocId": doc_id,
                "clusterId": "merge_intra_document_" + str(m),
                "title": title,
                "mentions": mentions,
            }
        else:
            new_cluster = {
                "clusterId": "merge_extra_document_" + str(m),
                "title": title,
                "mentions": mentions,
            }
        m += 1
        N_clusters.append(new_cluster)

    print("new clusters ", len(N_clusters), " cluster vecchi ", len(O_clusters))
    print("documento ", doc_id, "cluster nulli: ", p)
    N_clusters.extend(O_clusters)
    return N_clusters

def check_ground_truth(clusters: list, gt: dict) -> dict:
   
    t_positivo = 0
    f_positivo = 0
    f_negativo = 0

    titles_seen = set()

    for c in clusters:
        title = c.get("title")
        mentions = c.get("mentions", [])

        gt_mentions = gt.get(title)

        if gt_mentions is None:
            f_positivo += len(mentions)
            continue

        titles_seen.add(title)

        gt_keys = {(m.get("id"), m.get("start"), m.get("end")) for m in gt_mentions}
        cluster_keys = {(m.get("id"), m.get("start"), m.get("end")) for m in mentions}

        # true positive: presenti sia nel cluster che nella GT per questo titolo
        tp_keys = cluster_keys & gt_keys
        # false positive: nel cluster ma non attese per questo titolo
        fp_keys = cluster_keys - gt_keys
        # false negative: attese per questo titolo ma mancanti dal cluster
        fn_keys = gt_keys - cluster_keys

        t_positivo += len(tp_keys)
        f_positivo += len(fp_keys)
        f_negativo += len(fn_keys)

    # titoli presenti nella GT ma mai comparsi come titolo di alcun cluster nell'output:
    # tutte le loro menzioni attese sono false negative
    for title, gt_mentions in gt.items():
        if title not in titles_seen:
            f_negativo += len(gt_mentions)

    precision = t_positivo / (t_positivo + f_positivo) if (t_positivo + f_positivo) else 0.0
    recall = t_positivo / (t_positivo + f_negativo) if (t_positivo + f_negativo) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n--- Risultati ---")
    print(f"TP={t_positivo}  FP={f_positivo}  FN={f_negativo}")
    print(f"Precisione: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")


def run(input_path: str, output_path: str = OUTPUT_PATH):
    inizio = time.perf_counter()

    I = Path(input_path)
    O = Path(output_path)

    with Path(GT).open(encoding="utf-8") as f:
            gt: dict = json.load(f)[0]

    with I.open(encoding="utf-8") as f:
        clusters: list = json.load(f)

    Sep_D: dict[str, list] = separate_doc(clusters)

    Sep_C: dict[str, dict[int, list[tuple[str, str]]]] = {doc_id: separate_clusters(clusters) for doc_id, clusters in Sep_D.items()}
        # dict[doc_id, dict[numero_batch, list[tuple[id, clutser_title]]]]
    
    Merged_documents: list = []

    for doc_id, clusters_lists in Sep_C.items():
        print("incomicio lavoro su doc:", doc_id)
        workers = len(clusters_lists.keys())
        print("batch: ", workers)
        to_merge: list[list] = process_clusters_parallel(clusters_lists, workers)
        document_clusters = Sep_D[doc_id]
        merged_clusters : list = merge_clusters(document_clusters, to_merge, doc_id = doc_id)
        Merged_documents.extend(merged_clusters)
    
    Sep_A_C: dict[int, list[tuple[str, str]]] = separate_clusters(Merged_documents)
           # dict[batch_number, list[tuple[id, clutser_title]]]
    
    workers = len(Sep_A_C.keys())
    print("batch: ", workers)
    to_merge: list[list] = process_clusters_parallel(Sep_A_C, workers)
            # list[id_lists]
    Merged_A_Documents: list = merge_clusters(Merged_documents, to_merge, version = 1)

    Merged_All: list = Merged_A_Documents
    
    while to_merge != []:
        k = 2
        workers = workers // k
        new_Sep_A_C: dict[int, list[tuple[str, str]]] = separate_clusters(Merged_A_Documents, P = workers)
        print("batch: ", workers)
        to_merge: list[list] = process_clusters_parallel(new_Sep_A_C, workers)
        merged_A_clusters: list = merge_clusters(Merged_A_Documents, to_merge, version = 1)
        Merged_A_Documents = merged_A_clusters
        if workers == 1:
            to_merge.clear()
            Merged_All = Merged_A_Documents

    check_ground_truth(Merged_All, gt)

    with O.open("w", encoding="utf-8") as f:
        json.dump(Merged_All, f, ensure_ascii=False, indent=2)
    
    fine = time.perf_counter()
    print(f"Tempo di MERGE_CLUSTERS: {(fine - inizio)/60:.2f} minuti")

if __name__ == "__main__" :
    run(r"D:\Tesi PY\C_GROUND_TRUTH_F.json")
