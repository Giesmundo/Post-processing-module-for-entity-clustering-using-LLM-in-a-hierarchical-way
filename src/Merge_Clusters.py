from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict
from typing import Callable
from google import genai
import anthropic
import random
import tiktoken
from openai import OpenAI
from Coref_evaluation import (gt_json_to_clusters, system_json_to_clusters, evaluate_collection, evaluate_collection_er_metrics)
from data.yours_data import (OUTPUT_PATH_CLEAN, OUTPUT_PATH_MERGE, GOLD_CLUSTERS,ALREADY_MERGED_PATH, GENAI_MODELS, CLAUDE_MODELS, OPENAI_MODELS, 
                            API_KEY_GENAI, API_KEY_ANTHROPIC, API_KEY_OPENAI, MBS, BTC, LLM_MERGE)

OUTPUT_PATH = OUTPUT_PATH_MERGE
GT = GOLD_CLUSTERS
ERRORS_PATH = r"D:\Tesi PY\errors\errori_merge.json"

volta_lock = threading.Lock() #DEBUG
volta = 0  #DEBUG
m = 0
modello_corrente= 0
tot_I_tokens = 0
tot_O_tokens = 0
client_genai = genai.Client(api_key=API_KEY_GENAI)
client_claude = anthropic.Anthropic(api_key=API_KEY_ANTHROPIC)
client_openai = OpenAI(api_key=API_KEY_OPENAI)
_encoder = tiktoken.get_encoding("cl100k_base")
rate_limit_wait = 90.0
rate_limit_until = 0.0

prompt = """In Input hai una lista di titoli di Cluster, associati al loro id
es: [[id_es1, "Presidente del Consiglio"],[id_es2,"Marco Ramat"],[id_es3,"Francesco Cossiga"], [id_es4, "Primo Ministro"]]
crea un list[list] in cui ogn lista contiene gli ID dei cluster che riferiscono alla stessa entità.
Nella lista di output ci devono essere TUTTI gli elementi dati in input, se un cluster è l'unico a riferirsi ad un'entità allora mettilo singolo
esempio Output:
[[ides_1, ides_45], [ides_4, ides_7, ides_89], [ides_76], [ides_21]]
Restituisci ESCLUSIVAMENTE LA LISTA DI LISTE, nessun testo prima o dopo, nessun esempio, nessun markdown.
ATTENZIONE A RIPORTARE GLI ID DEI CLUSTER ESATTAMENTE IDENTICI, NON CI DEVONO ESSERE ERRORI NE DUPLICATI.
Input:
{}"""

def count_tokens(text: str) -> int:

    return len(_encoder.encode(text))

def wait_if_rate_limited():

    while True:
        with volta_lock:
            remaining = rate_limit_until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))

def signal_rate_limit(seconds: float = rate_limit_until):

    global rate_limit_until

    with volta_lock:
        x = time.time() + seconds
        if x > rate_limit_until:
            rate_limit_until = x

def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento

    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    return doc_clusters

def separate_clusters(clusters: list, batch_length: int, P: int = 0) -> dict: # separa i singoli cluster

    titles: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if P == 0:
        n = 0
        for cluster in clusters:
            if cluster is None:
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            if id is None or title is None:
                continue
            id_sliced = id[-20:]
            if len(titles[n]) >= batch_length:
                titles[n + 1].append((id_sliced, title))
                random.shuffle(titles[n + 1])
                n += 1
            else:
                titles[n].append((id_sliced, title))
                random.shuffle(titles[n])

    elif P == -1:
        P = 10
        titles = {i: [] for i in range(P)}
        A_groups : dict[str, list] = defaultdict(list)
        for cluster in clusters:
            if cluster is None:
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            if id is None or title is None:
                continue
            id_sliced = id[-20:]
            first_char = title[0].lower() if title[0].isalpha() else "#"
            A_groups[first_char].append((id_sliced, title))
            
        for _, items in A_groups.items():
            smallest_group = min(titles, key=lambda k: len(titles[k]))
            titles[smallest_group].extend(items)
    else:
        P -= 1
        n = 0
        for cluster in clusters:
            if cluster is None:
                continue
            title = cluster.get("title")
            id = cluster.get("clusterId")
            if id is None or title is None:
                continue
            id_sliced = id[-20:]
            if len(titles[n]) >= batch_length:
                if n == P:
                    n = -1
                titles[n + 1].append((id_sliced, title))
                random.shuffle(titles[n + 1])
                n += 1
            else:
                titles[n].append((id_sliced, title))
                random.shuffle(titles[n])
    return titles

def process_clusters_parallel(Sep_C: dict[int, list[tuple[str, str]]], max_workers: int, LLM: Callable) -> list[list]:

    to_merge: list[list] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {}
        for batch_number, cluster_list in Sep_C.items():
            future_obj = executor.submit(LLM, cluster_list, prompt)
            tasks[future_obj] = (batch_number)

        for task in as_completed(tasks):
            try:
                result = task.result()
                to_merge.extend(result)
            except Exception as e:
                print("Errore nella chiamata al modello nel batch: ", tasks[task], " - ", e)

    return to_merge

def A_call_llm(cluster_list: list, prompt: str, retry: int = 2, delay: float = 5.0) -> dict:

    global tot_O_tokens
    global tot_I_tokens
    global volta

    n = volta + 1
    with volta_lock:
        volta += 1
    print("chiamata: ", n)
    Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
    I_tokens = count_tokens(Prompt)
    with volta_lock:
        tot_I_tokens += I_tokens
    
    for model in CLAUDE_MODELS:
        for attempt in range(retry):
            wait_if_rate_limited()
            try:
                with client_claude.messages.stream(
                    model=model,
                    max_tokens=64000,
                    messages=[{"role": "user", "content": Prompt}]
                ) as stream:
                    for _ in stream.text_stream:
                        pass
                    final_message = stream.get_final_message()
                text = final_message.content[0].text.strip()
                O_tokens = count_tokens(text)
                with volta_lock:
                    tot_O_tokens += O_tokens
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("Risposta senza JSON valido.")
                Result = json.loads(text)
                return Result

            except Exception as e:
                err = str(e)
                print(err)
                if "529" in err or "overloaded" in err.lower():
                    print(model + " sovraccarico")
                    break
                if "rate" in err.lower() or "429" in err:
                    print(model + " rate limit")
                    signal_rate_limit()
                    break
                if attempt < retry-1:
                    print(f"chiamata {n} nuovo tentativo")
                    time.sleep(delay)
                else:
                    print(f"Errore persistente con {model}: {e}")
                    break

    raise RuntimeError("Tutti i modelli hanno esaurito la quota.")

def O_call_llm(cluster_list: list, prompt: str, retry: int = 4, delay: float = 5.0) -> dict:

    global tot_O_tokens
    global tot_I_tokens
    global volta

    n = volta + 1
    with volta_lock:
        volta += 1
    print("chiamata: ", n)
    Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
    I_tokens = count_tokens(Prompt)
    with volta_lock:
        tot_I_tokens += I_tokens

    for model in OPENAI_MODELS:
        for attempt in range(retry):
            wait_if_rate_limited()
            text = None
            try:
                text_chunks = []
                with client_openai.with_options(timeout=300.0).chat.completions.stream(
                    model=model,
                    max_completion_tokens=128000,
                    messages=[{"role": "user", "content": Prompt}]
                ) as stream:
                    for event in stream:
                        if event.type == "content.delta":
                            text_chunks.append(event.delta)
                    final_completion = stream.get_final_completion()

                text = "".join(text_chunks).strip() if text_chunks else final_completion.choices[0].message.content.strip()
                O_tokens = count_tokens(text)
                with volta_lock:
                    tot_O_tokens += O_tokens

                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("Risposta senza JSON valido.")
                Result = json.loads(text)
                return Result

            except Exception as e:
                err = str(e)
                print(err)
                if "503" in err or "overloaded" in err.lower() or "service_unavailable" in err.lower():
                    print(model + " sovraccarico")
                    break
                if "rate" in err.lower() or "429" in err or "rate_limit_exceeded" in err.lower():
                    print(model + " rate limit")
                    signal_rate_limit()
                    break
                if attempt < retry - 1:
                    print(f"chiamata {n} nuovo tentativo")
                    time.sleep(delay)
                else:
                    print(f"Errore persistente con {model}: {e}")
                    break

    raise RuntimeError("Tutti i modelli hanno esaurito la quota.")

def G_call_llm(cluster_list: list, prompt: str, retry: int = 5, delay: float = 5.0) -> str:

    global tot_O_tokens
    global tot_I_tokens
    global modello_corrente

    for i in range(modello_corrente, len(GENAI_MODELS)):
        modello = GENAI_MODELS[i]
        for attempt in range(retry):
            try:
                Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
                I_tokens = count_tokens(Prompt)
                with volta_lock:
                    tot_I_tokens += I_tokens
                response = client_genai.models.generate_content(model=modello, contents=Prompt)
                text = response.text.strip()
                O_tokens = count_tokens(text)
                with volta_lock:
                    tot_O_tokens += O_tokens
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
            if x[-20:] == id:
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

    N_clusters.extend(O_clusters)
    N_clusters = [c for c in N_clusters if c is not None and c.get("mentions")]

    return N_clusters

def checkifall(to_merge: list[list], clusters: dict[int, list[tuple[str, str]]]) -> list[list]:

    ids2 = {
        coppia[0]
        for cluster_list in clusters.values()
        for coppia in cluster_list
    }

    to_merge = [
        [id for id in id_list if id in ids2]
        for id_list in to_merge
    ]
    to_merge = [id_list for id_list in to_merge if id_list]

    ids = {
        id
        for id_list in to_merge
        for id in id_list
    }

    seen = set()
    deduped = []
    for id_list in to_merge:
        new_list = []
        for id in id_list:
            if id not in seen:
                new_list.append(id)
                seen.add(id)
        if new_list:
            deduped.append(new_list)
    to_merge = deduped

    for cluster_list in clusters.values():
        for coppia in cluster_list:
            if coppia[0] not in ids:
                to_merge.append([coppia[0]])
                ids.add(coppia[0])

    return to_merge   

def run(input_path: str, batch_to_be_combined: int, batch_length: int, llm: str, output_path: str = OUTPUT_PATH):
    inizio = time.perf_counter()

    global tot_I_tokens
    global tot_O_tokens
    I = Path(input_path)
    O = Path(output_path)

    if llm == "anthropic":
        LLM_f = A_call_llm
    elif llm == "openai":
        LLM_f = O_call_llm
    else:
        LLM_f = G_call_llm

    with I.open(encoding="utf-8") as f:
        clusters: list = json.load(f)

    Sep_D: dict[str, list] = separate_doc(clusters)
    
    Merged_documents: list = []

    if len(Sep_D.keys()) > 1:

        Sep_C: dict[str, dict[int, list[tuple[str, str]]]] = {doc_id: separate_clusters(clusters, batch_length) for doc_id, clusters in Sep_D.items()}
             # dict[doc_id, dict[numero_batch, list[tuple[id, clutser_title]]]]

        for doc_id, clusters_lists in Sep_C.items():
            print("incomicio lavoro su doc:", doc_id)
            workers = len(clusters_lists.keys())
            print("batch: ", workers)
            if workers == 1:
                to_merge: list[list] = process_clusters_parallel(clusters_lists, workers, LLM_f)
                to_merge = checkifall(to_merge, clusters_lists)
                document_clusters = Sep_D[doc_id]
                merged_clusters : list = merge_clusters(document_clusters, to_merge, doc_id = doc_id)
                Merged_documents.extend(merged_clusters)
            else:
                current_clusters = Sep_D[doc_id]
                while True:
                    to_merge = process_clusters_parallel(clusters_lists, workers, LLM_f)
                    to_merge = checkifall(to_merge, clusters_lists)
                    merged_clusters = merge_clusters(current_clusters, to_merge, doc_id=doc_id)
                    current_clusters = merged_clusters
                    if workers == 1:
                        Merged_documents.extend(merged_clusters)
                        break
                    workers = max(workers // batch_to_be_combined, 1)
                    random.shuffle(merged_clusters)
                    clusters_lists = separate_clusters(merged_clusters, batch_length, P=workers)
    else:
        Merged_documents = clusters

    random.shuffle(Merged_documents)

    Sep_A_C: dict[int, list[tuple[str, str]]] = separate_clusters(Merged_documents, batch_length)
        # dict[batch_number, list[tuple[id, clutser_title]]]
    
    workers = len(Sep_A_C.keys())
    print("batch/workers: ", workers)
    to_merge: list[list] = process_clusters_parallel(Sep_A_C, workers, LLM_f)
            # list[id_lists]
    to_merge = checkifall(to_merge, Sep_A_C)
    Merged_A_Documents: list = merge_clusters(Merged_documents, to_merge, version = 1)

    Merged_All: list = Merged_A_Documents

    deepness = 0
    while workers >= 1:
        if deepness == 4 and workers > 3:
            break
        deepness += 1
        workers = max(workers // batch_to_be_combined, 1)
        print("workers: ", workers)
        new_Sep_A_C: dict[int, list[tuple[str, str]]] = separate_clusters(Merged_A_Documents, batch_length, P = workers)
        print("batch: ", len(new_Sep_A_C.keys()))
        to_merge: list[list] = process_clusters_parallel(new_Sep_A_C, workers, LLM_f)
        to_merge = checkifall(to_merge, new_Sep_A_C)
        merged_A_clusters: list = merge_clusters(Merged_A_Documents, to_merge, version = 1)
        Merged_A_Documents = merged_A_clusters
        random.shuffle(Merged_A_Documents)
        if workers == 1:
            Prompt = prompt.replace("{}", json.dumps(new_Sep_A_C[0], ensure_ascii=False, indent=2), 1)
            print("Tokens last batch: ", count_tokens(Prompt))
            print("singolo gruppo raggiunto")
            Merged_All = Merged_A_Documents
            break

    print("profondità del merge: ", deepness)

    if workers > 1:
        Merged_A_Documents = [c for c in Merged_A_Documents if c is not None]
        Merged_A_Documents.sort(key=lambda c: c.get("title") or "")
        new_Sep_A_C : dict[int, list[tuple[str, str]]] = separate_clusters(Merged_A_Documents, batch_length=0, P = -1)
        new_Sep_A_C = {k: v for k, v in new_Sep_A_C.items() if v}
        if new_Sep_A_C:
            first_key = next(iter(new_Sep_A_C))
            Prompt = prompt.replace("{}", json.dumps(new_Sep_A_C[first_key], ensure_ascii=False, indent=2), 1)
            print("Tokens last batch: ", count_tokens(Prompt))
            to_merge = process_clusters_parallel(new_Sep_A_C, max_workers=len(new_Sep_A_C.keys()), LLM=LLM_f)
            to_merge = checkifall(to_merge, new_Sep_A_C)
        else:
            to_merge = []
        merged_A_clusters : list = merge_clusters(Merged_A_Documents, to_merge, version = 1)
        Merged_All = merged_A_clusters

    if ALREADY_MERGED_PATH:
        with Path(ALREADY_MERGED_PATH).open(encoding="utf-8") as f:
            already_merged: list = json.load(f)
        Merged_All.extend(already_merged)
        Merged_All = [c for c in Merged_All if c is not None]
        Merged_All.sort(key=lambda c: c.get("title") or "")
        new_Sep_A_C : dict[int, list[tuple[str, str]]] = separate_clusters(Merged_All, batch_length=0, P = -1)
        new_Sep_A_C = {k: v for k, v in new_Sep_A_C.items() if v}
        if new_Sep_A_C:
            to_merge = process_clusters_parallel(new_Sep_A_C, max_workers=len(new_Sep_A_C.keys()), LLM=LLM_f)
            to_merge = checkifall(to_merge, new_Sep_A_C)
        else:
            to_merge = []
        merged_A_clusters : list = merge_clusters(Merged_All, to_merge, version = 1)
        Merged_All = merged_A_clusters

    with O.open("w", encoding="utf-8") as f:
        json.dump(Merged_All, f, ensure_ascii=False, indent=2)

    try:
        gold_clusters = gt_json_to_clusters(GT)
        sys_clusters = system_json_to_clusters(O)
        conll_results = evaluate_collection(gold_clusters, sys_clusters)
        er_results = evaluate_collection_er_metrics(gold_clusters, sys_clusters)
        print("\n=== Famiglia CoNLL / CDCR ===")
        print("MUC   : P={:.4f}  R={:.4f}  F1={:.4f}".format(
            conll_results["muc"]["precision"], conll_results["muc"]["recall"], conll_results["muc"]["f1"]))
        print("B3    : P={:.4f}  R={:.4f}  F1={:.4f}".format(
            conll_results["b3"]["precision"], conll_results["b3"]["recall"], conll_results["b3"]["f1"]))
        print("CEAFe : P={:.4f}  R={:.4f}  F1={:.4f}".format(
            conll_results["ceafe"]["precision"], conll_results["ceafe"]["recall"], conll_results["ceafe"]["f1"]))
        print("CoNLL-F1: {:.4f}".format(conll_results["conll_f1"]))

        print("\n=== Famiglia ER / clustering ===")
        print("ACC   : {:.4f}".format(er_results["acc"]))
        print("Purity: {:.4f}  Inverse-Purity: {:.4f}  FP: {:.4f}".format(
            er_results["purity"], er_results["inverse_purity"], er_results["fp_measure"]))
        print("NMI   : {:.4f}  ARI: {:.4f}".format(er_results["nmi"], er_results["ari"]))

    except Exception as e:
        print("Errore nella valutazione delle metriche di coreferenza.")
        print(e)

    print("\n")
    print("token in Input: ", tot_I_tokens)
    print("token in Output: ", tot_O_tokens)

    fine = time.perf_counter()
    print(f"Tempo di MERGE_CLUSTERS: {(fine - inizio)/60:.2f} minuti")

if __name__ == "__main__" :
    run(OUTPUT_PATH_CLEAN, batch_to_be_combined = BTC, batch_length = MBS, output_path=OUTPUT_PATH_MERGE, llm = LLM_MERGE)