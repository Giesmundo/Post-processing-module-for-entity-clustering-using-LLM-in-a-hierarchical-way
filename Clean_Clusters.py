from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
from pathlib import Path
from collections import defaultdict
from google import genai
from openai import OpenAI

volta_lock = threading.Lock() #DEBUG
volta = 0                     #DEBUG

INPUT_PATH  = r"INPUT_FILE"
OUTPUT_PATH = r"OUTPUT_FILE"
API_KEY = "GENAI_KEY"
MODELLO1 =  "gemini-3.5-flash"
MODELLO2 = "gemini-3.1-flash-lite"
MODELLO3 =  "gemini-3-flash-preview"

KEY = "OPENAI_KEY"

MODELS = [MODELLO1, MODELLO2, MODELLO3]

prompt = """Ricevi in input un dizionario JSON con nel formato "id": [titolo_cluster, testo_menzione, contesto_menzione].

Crea un oggetto json valido con chiave id e valore un Boolean o null seguendo queste regole:

Per ogni valore [titolo_cluster, testo_menzione, contesto_menzione]:
- Il campo "contesto" contiene le parole prima e dopo la menzione nel testo originale, compreso testo_menzione.

Assegna il valore seguendo queste regole in ordine:
1. Se il testo_menzione NON è presente in contesto_menzione, oppure è sottostringa di un'altra parola del contesto_menzione (es. MAGGIO in MAGGIORANZA) assegna null
2. Se testo_menzione e titolo_cluster riferiscono alla stessa entità, verifica analizzando contesto_menzione, assegna True
3. Altrimenti assegna False

Per la regola 2, considera:
- Sinonimi, nomignoli, soprannomi, abbreviazioni
- Nomi parziali (es. "Mario" per "Mario Rossi")
- Qualsiasi altro riferimento indiretto alla stessa entità, specialmente per persone

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido.
- Usa come chiave SOLO l'id
- Esempio: ("329": true, "412": false, "87": null)
- Includi TUTTE le chiavi, nessuna esclusa
- Nessun testo aggiuntivo, nessun markdown, nessuna spiegazione

Input:
{}"""

prompt2 = """"""

model_curr = 0
volta = 0                     #DEBUG
client = genai.Client(api_key=API_KEY)
client2 = OpenAI(api_key=KEY)

def separate_doc(clusters: list) -> dict[str, list]:   # separa i cluster per doumento
    doc_clusters: dict[str, list] = defaultdict(list)
    for c in clusters:
        doc_id = c.get("originalDocId")
        doc_clusters[doc_id].append(c)

    return doc_clusters

def separate_mentions(clusters: list) -> dict[int, tuple[str, str, str]]: # separa le singole menzioni
    
    mentions: dict[int, tuple[str, str, str]] = defaultdict(tuple)

    for cluster in clusters:
        title = cluster.get("title")
        for mention in cluster.get("mentions"):
            if mention is None:
                    continue
            id = mention.get("id")
            text = mention.get("text") 
            context = mention.get("context")
            mentions[id] = (title, text, context)
    return mentions

def process_Mentions_parallel(Sep_M: dict[str, dict[int, tuple[str, str, str]]],  # invia al LLM i gruppi di menzioni in parallelo
                              max_workers: int = 8) -> dict[str, dict[int, bool]]: # e ritorna le menzioni con valori aggiornati

    def chunked(d: dict):   # divide la lista di menzioni in chunk da 70 menzioni l'uno
        items = list(d.items())
        size = 70
        n = (len(items) // size)
        chunks = []
        if n > 0:
            for i in range(n - 1):
                chunks.append(dict(items[i*size : (i+1)*size]))
            if (len(items) % size) > (size / 2):
                chunks.append(dict(items[(n-1)*size : n*size]))
                chunks.append(dict(items[n*size:]))
            else:
                chunks.append(dict(items[(n-1)*size:]))
            return chunks
        else:
            chunks.append(dict(items))
            return chunks

    Checked_mentions = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {}
        for doc_id, mention_list in Sep_M.items():
            print("documento: " + str(doc_id) + " di " + str(len(mention_list)) + " menzioni") #DEBUG
            chunks = chunked(mention_list)
            for chunk_id, chunk in enumerate(chunks):
                print("chunk: " + str(chunk_id) + " di " + str(len(chunk)) + " menzioni") #DEBUG
                future_obj = executor.submit(call_llm, chunk, prompt)
                tasks[future_obj] = (doc_id, chunk_id)

        partial_results: dict[str, list[tuple[int, dict]]] = defaultdict(list)

        for task in as_completed(tasks):
            doc_id, chunk_id = tasks[task]
            try:
                result = task.result()
                partial_results[doc_id].append((chunk_id, result))
            except Exception as e:
                print(f"Errore nel documento {doc_id}, chunk {chunk_id}: {e}")

    for doc_id, chunks_results in partial_results.items():
        chunks_results.sort(key=lambda x: x[0])
        merged = {}
        for _, chunk_result in chunks_results:
            merged.update(chunk_result)
        Checked_mentions[doc_id] = merged

    return Checked_mentions

def call_llm(diz: dict, prompt: str, retry: int = 20, delay: float = 5.0) -> dict[int, bool]: # chimata a LLM

    global volta              #DEBUG
    with volta_lock:          #DEBUG
        volta += 1            #DEBUG
        n = volta             #DEBUG

    diz_ = {
            str(k): [v[0], v[1], v[2]]
            for k, v in diz.items()
            }
    Prompt = prompt.replace("{}", json.dumps(diz_, ensure_ascii=False, indent=2), 1)
    print("chiamata modello per menzioni n. " + str(n)) #DEBUG  
    for model in MODELS:
        for attempt in range(retry):
            try:
                #response = client.responses.create({
                    #"model": "gpt-5.5",
                    #"input": Prompt
                #})
                response = client.models.generate_content(model=model, contents=Prompt)
                text = response.text.strip()
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    text = text[start:end+1]
                else:
                    raise ValueError("La risposta del modello non contiene un JSON valido.")
                result_ = json.loads(text)
                Result = {}
                for key, value in result_.items():
                    Result[int(key)] = value
                print("fine chiamata n. " + str(n)) #DEBUG
                return Result
            
            except Exception as e:
                err = str(e)
                if "429" in err:                                      
                    print(model + " esaurito")               #DEBUG
                    break
                if attempt < retry - 1:
                    time.sleep(delay)
                else:
                    print(f"Errore persistente con {model}: {e}")       #DEBUG
                    break

    raise RuntimeError("Tutti i modelli hanno esaurito la quota.")

def clean_clusters(clusters: dict[str, list], C_mentions: dict[str, dict[int, bool]]) -> list: # pulisce i cluster

    for doc_id, cluster_list in clusters.items():
        orphans = []
        try:
            mentions_dict = C_mentions[doc_id]
        except:
            continue
        for cluster in cluster_list:
            T_mentions = []
            for mention in cluster.get("mentions"):
                if mention is None:
                    continue
                mention_id = mention.get("id")
                try:
                    mention_value = mentions_dict[(mention_id)]
                except:
                    print("errore nel documento:", doc_id)
                    print("mention_id:", mention_id)
                    raise KeyError("Menzione non trovata")
                if mention_value is None:
                    print("MENZIONE NULLA", mention_id, "documento:", doc_id)   #DEBUG
                    continue
                else:
                    if not mention_value:
                        mention_text = mention.get("text")
                        cluster_title = cluster.get("title")
                        source_type = cluster.get("type")
                        id_cluster = cluster.get("clusterId")
                        print("testo: " + mention_text + " id: " + str(mention_id) + " Cluster: " + cluster_title)   #DEBUG
                        new_cluster = {
                            "originalDocId": doc_id,
                            "clusterId": id_cluster + "_orphan_" + mention_text,
                            "title": mention_text,
                            "type": source_type,
                            "mentions": [mention],
                        }
                        orphans.append(new_cluster)
                    else:
                        T_mentions.append(mention)
            cluster["mentions"] = T_mentions
        clusters[doc_id].extend(orphans)
        print(f"documento pulito: {doc_id}") #DEBUG

    return [c for c in clusters.values()]

def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    I = Path(input_path)
    O = Path(output_path)

    with I.open(encoding="utf-8") as f:
        clusters: list= json.load(f)

    Sep_D: dict[str, list] = separate_doc(clusters)

    Sep_M: dict[str, dict[int, tuple[str, str, str]]] = {doc_id: separate_mentions(clusters) for doc_id, clusters in Sep_D.items()}
    
    all_Checked_mentions: dict[str, dict[int, bool]] = process_Mentions_parallel(Sep_M)

    all_Cleaned_clusters: list = clean_clusters(Sep_D, all_Checked_mentions)
    
    with O.open("w", encoding="utf-8") as f:
        json.dump(all_Cleaned_clusters, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    inizio = time.perf_counter()
    run()
    fine = time.perf_counter()
    print(f"Tempo di esecuzione: {(fine - inizio)/60:.2f} minuti")
