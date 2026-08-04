from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
from pathlib import Path
from collections import defaultdict
import anthropic
from google import genai
import tiktoken

volta_lock = threading.Lock() #DEBUG
volta = 0                     #DEBUG

OUTPUT_PATH = r"D:\Tesi PY\output\Clusters_Puliti.json"
GROUND_TRUTH = r"D:\Tesi PY\Ground_truth\C_GROUND_TRUTH_O.json"
ERRORS_PATH = r"D:\Tesi PY\errors\errori.txt"
MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]
CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"]  # dal più economico al più capace

prompt = """I tuo compito è quello di verificare che due stringhe siano riferite ad una stessa entità, usando le parole prime e dopo la seconda stringa.
In Input hai una lista di Cluster, e le menzioni di ogni clutser, cioè le ricorrenze dell'entità del cluster:
verifica l'appartenenza di ogni menzione comparando title dei cluster e text delle menzioni
usando anche, se c'è, il context delle menzioni che presenta le parole prima e dopo al text.
In output crea un dict(str, list), la lista di interi, e inserisci:
in ELIMINARE le menzioni il cui text NON è presente nel proprio context, se c'è. 
in SPOSTARE le menzioni il cui text è giusto ma NON si rifersice al cluster in cui si trova.
OUTPUT es:
{"ELIMINARE":[ID_ES1, ID_ES2], "SPOSTARE": [ID_ES3, ID_ES4]}
Rispondi SOLO CON JSON VALIDO, senza testo prima o dopo, senza esempi, senza markdown.
Input:
{}"""

Tot_Predette = 0
Tot_Corrette = 0
Tot_Gt = 0
modello_corrente = 0
tot_I_tokens = 0
tot_O_tokens = 0
client = genai.Client(api_key="")
client_claude = anthropic.Anthropic(api_key="")
_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))

def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento

    print("inizio separate_doc")

    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    print("fine separate_doc")

    return doc_clusters

def separate_clusters(clusters: list, batch_length: int) -> dict[int, list]: # separa le singole menzioni

    mentions: dict[int, list] = defaultdict(list)
    n = 0
    for cluster in clusters:
        if len(cluster) > batch_length - len(mentions[n]):
            n += 1
            mentions[n].append(cluster)
        else:
            mentions[n].append(cluster)
    print("numero di batch: ", n)
    return mentions

def process_clusters_parallel(Sep_C: dict[str, list], max_workers: int) -> dict[str, list]:

    print("inizio process_cluster_parallel")

    to_change_ids: dict[str, list] = {"SPOSTARE": [], "ELIMINARE": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {}
        for S_type, cluster_list in Sep_C.items():
            future_obj = executor.submit(A_call_llm, cluster_list, prompt)
            tasks[future_obj] = (S_type)

        for task in as_completed(tasks):
            try:
                result = task.result()
                to_change_ids["SPOSTARE"].extend(result["SPOSTARE"])
                to_change_ids["ELIMINARE"].extend(result["ELIMINARE"])
            except Exception as e:
                print("Errore nella chiamata al modello per il tipo:", tasks[task], "-", e)
    print("fine process_cluster_parallel")
    return to_change_ids

def A_call_llm(cluster_list: list, prompt: str, retry: int = 10, delay: float = 5.0) -> dict[str, list]: # chimata a LLM

    global tot_O_tokens
    global tot_I_tokens
    global volta              #DEBUG
    with volta_lock:          #DEBUG
        volta += 1            #DEBUG
        n = volta             #DEBUG

    Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
    print("chiamata modello per menzioni n. " + str(n)) #DEBUG 
    I_tokens = count_tokens(Prompt)
    with volta_lock:
        tot_I_tokens += I_tokens

    for model in CLAUDE_MODELS:
        for attempt in range(retry):
            try:
                print(f"chiamata modello: {model} tentativo: {attempt + 1}")
                response = client_claude.messages.create(
                        model=model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": Prompt}]
                    )
                text = response.content[0].text.strip()
                O_tokens = count_tokens(text)
                with volta_lock:
                    tot_O_tokens += O_tokens
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("La risposta del modello non contiene un JSON valido.")
                result_ = json.loads(text)

                Result = {
                    "SPOSTARE": [int(x) for x in result_.get("SPOSTARE", [])],
                    "ELIMINARE": [int(y) for y in result_.get("ELIMINARE", [])]
                }
                return Result
            
            except Exception as e:
                    err = str(e)
                    print(f"ERRORE COMPLETO: {repr(e)}")
                    print("RISPOSTA GREZZA: ")
                    print(text)  
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

def G_call_llm(cluster_list: list, prompt: str, retry: int = 10, delay: float = 5.0) -> dict[str, list]:

    global tot_I_tokens
    global tot_O_tokens
    global modello_corrente
    for modello in MODELS:
        for attempt in range(retry):
            try:
                Prompt = prompt.replace("{}", json.dumps(cluster_list, ensure_ascii=False, indent=2), 1)
                I_tokens = count_tokens(Prompt)
                with volta_lock:
                    tot_I_tokens += I_tokens
                response = client.models.generate_content(model=modello, contents=Prompt)
                text = response.text.strip()
                O_tokens = count_tokens(text)
                with volta_lock:
                    tot_O_tokens += O_tokens
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("La risposta del modello non contiene un JSON valido.")
                result_ = json.loads(text)

                Result = {
                    "SPOSTARE": [int(x) for x in result_.get("SPOSTARE", [])],
                    "ELIMINARE": [int(y) for y in result_.get("ELIMINARE", [])]
                }
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

def clean_clusters(cluster_list: list, to_change: dict[str, list], NER_type: bool) -> list: # pulisce i cluster
    
    print("inizio clean_cluster")

    sposta_ids = set(to_change.get("SPOSTARE", []))
    elimina_ids = set(to_change.get("ELIMINARE", []))

    N_clusters = []
    n = 0
    for cluster in cluster_list:
        T_mentions = []
        for mention in cluster.get("mentions"):
            if mention is None:
                continue
            mention_id = mention.get("id")
            if mention_id in elimina_ids:
                print("ELIMINATA: ", mention_id, " Cluster: ", cluster.get("title"), " Menzione: ", mention.get("text"))  # DEBUG
                continue
            if mention_id in sposta_ids:
                print("SPOSTATA: ", mention_id, " Cluster: ", cluster.get("title"), " Menzione: ", mention.get("text"))
                id_cluster = cluster.get("clusterId")
                doc_id = cluster.get("originalDocId")
                mention_text = mention.get("text")
                if NER_type:
                    source_type = cluster.get("type")
                    new_cluster = {
                        "originalDocId": doc_id,
                        "clusterId": str(id_cluster) + "_orphan_" + str(n),
                        "title": mention_text,
                        "type": source_type,
                        "mentions": [mention],
                    }
                else:
                    new_cluster = {
                        "originalDocId": doc_id,
                        "clusterId": str(id_cluster) + "_orphan_" + str(n),
                        "title": mention_text,
                        "mentions": [mention],
                    }
                N_clusters.append(new_cluster)
                n += 1
                continue
            T_mentions.append(mention)
        cluster["mentions"] = T_mentions
    cluster_list.extend(N_clusters)
    print("fine clean_cluster")
    return cluster_list

def check_ground_truth(doc_id: str, to_change: dict[str, list], gt: dict):

    global Tot_Predette
    global Tot_Corrette
    global Tot_Gt
    print("inizio check_ground_truth")
    with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
        f.write(f"Errori per documento: {doc_id}\n")
    falso_positivo_E = 0
    vero_positivo_E = 0
    falso_negativo_E = 0
    falso_positivo_S = 0
    vero_positivo_S = 0
    falso_negativo_S = 0

    gt_elimina = [int(k) for k, v in gt.get(doc_id, {}).items() if v == "ELIMINARE"]
    gt_sposta = [int(k) for k, v in gt.get(doc_id, {}).items() if v == "SPOSTARE"]
    sposta_ids = set(to_change.get("SPOSTARE", []))
    elimina_ids = set(to_change.get("ELIMINARE", []))

    for e_ids in elimina_ids:
        if e_ids in gt_elimina:
            vero_positivo_E += 1
        else:
            falso_positivo_E += 1
            with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
                f.write(f"Errore ELIMINARE: menzione {e_ids} non era da eliminare.\n")

    for s_ids in sposta_ids:
        if s_ids in gt_sposta:
            vero_positivo_S += 1
        else:
            falso_positivo_S += 1
            with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
                f.write(f"Errore SPOSTARE: menzione {s_ids} non era da spostare.\n")

    falso_negativo_E = len(gt_elimina) - vero_positivo_E
    falso_negativo_S = len(gt_sposta) - vero_positivo_S

    for e_ids in gt_elimina:
        if e_ids not in elimina_ids:
            with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
                f.write(f"Errore ELIMINARE: menzione {e_ids} era da eliminare ma non è stata eliminata.\n")

    for s_ids in gt_sposta:
        if s_ids not in sposta_ids:
            with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
                f.write(f"Errore SPOSTARE: menzione {s_ids} era da spostare ma non è stata spostata.\n")

    print("menzioni spostate correttamente: ", vero_positivo_S, " su", len(gt_sposta))
    print("menzioni eliminate correttamente: ", vero_positivo_E, " su", len(gt_elimina))
    print("menzioni spostate erroneamente: ", falso_positivo_S)
    print("menzioni eliminate erroneamente: ", falso_positivo_E)
    print("menzioni non spostate: ", falso_negativo_S)
    print("menzioni non eliminate: ", falso_negativo_E)
    tot_predette = vero_positivo_S + vero_positivo_E + falso_positivo_S + falso_positivo_E
    tot_gt = len(gt_sposta) + len(gt_elimina)
    tot_corrette = vero_positivo_S + vero_positivo_E
    Tot_Corrette += tot_corrette
    Tot_Predette += tot_predette
    Tot_Gt += tot_gt
    precision = (tot_corrette / tot_predette * 100) if tot_predette > 0 else 0
    recall = (tot_corrette / tot_gt * 100) if tot_gt > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print("Precisione (corrette su predette):", precision, " %")
    print("Recall (corrette su totale GT): ", recall, " %")
    print("F1 score: ", f1, " %")

def run(input_path: str, context: bool, NER_type: bool, batch_length: int, output_path: str = OUTPUT_PATH):
    inizio = time.perf_counter()

    global Tot_Corrette
    global Tot_Gt
    global Tot_Predette
    global tot_O_tokens
    global tot_I_tokens

    I = Path(input_path)
    O = Path(output_path)

    with I.open(encoding="utf-8") as f:
        clusters: list = json.load(f)

    with Path(GROUND_TRUTH).open(encoding="utf-8") as f:
        gt: dict = json.load(f)

    Sep_D: dict[str, list] = separate_doc(clusters)

    Sep_C: dict[str, dict[int, list]] = {doc_id: separate_clusters(clusters, batch_length) for doc_id, clusters in Sep_D.items()}

    if not(context):
        for doc_id, batches in Sep_C.items():
            for _, clusters_list in batches.items():
                for cluster in clusters_list:
                    for mention in cluster["mentions"]:
                        del mention["context"]

    if not(NER_type):
        for doc_id, batches in Sep_C.items():
            for _, clusters_list in batches.items():
                for cluster in clusters_list:
                    del cluster["type"]
                    
    Cleaned_documents: list = []

    with Path(ERRORS_PATH).open("a", encoding="utf-8") as f:
        f.write("-------------------------OPERAZIONE DI PULIZIA--------------------------\n")

    for doc_id, clusters_lists in Sep_C.items():
        print("incomicio lavoro su doc:", doc_id)
        workers = len(clusters_lists)
        to_change: dict[str, list] = process_clusters_parallel(clusters_lists, workers)

        document_clusters = Sep_D[doc_id]

        check_ground_truth(doc_id, to_change, gt)

        Cleaned_clusters = clean_clusters(document_clusters, to_change, NER_type)

        Cleaned_documents.extend(Cleaned_clusters)

    precision = (Tot_Corrette / Tot_Predette * 100) if Tot_Predette > 0 else 0
    recall = (Tot_Corrette / Tot_Gt * 100) if Tot_Gt > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    print("Precisione totale corpus:", precision, " %")
    print("Recall totale corpus: ", recall, " %")
    print("F1 totale corpus: ", f1, " %")
    print("token in Input: ", tot_I_tokens)
    print("token in Output: ", tot_O_tokens)

    with O.open("w", encoding="utf-8") as f:
        json.dump(Cleaned_documents, f, ensure_ascii=False, indent=2)
    
    fine = time.perf_counter()
    print(f"Tempo di CLEAN_CLUSTERS: {(fine - inizio)/60:.2f} minuti")

    return O