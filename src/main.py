import Clean_Clusters as Clean_Clusters
import Merge_Clusters as Merge_Clusters
import time
from data.yours_data import (INPUT_PATH, OUTPUT_PATH_CLEAN, OUTPUT_PATH_MERGE, CX, NT, CBS, MBS, BTC, LLM_CLEAN, LLM_MERGE)

inizio = time.perf_counter()
with open(r"D:\Tesi PY\errors\errori.txt", "w") as f:
    f.write("")
print("----------------Inizio main-----------------------")
print("----------------INIZIO CLEAN_CLUSTERS-----------------")
Clean_Clusters.run(INPUT_PATH, context = CX, NER_type = NT, batch_length = CBS, output_path=OUTPUT_PATH_CLEAN, llm = LLM_CLEAN)
print("----------------FINITO CLEAN_CLUSTERS-------------")
print("----------------INIZIO MERGE_CLUSTERS-------------")
Merge_Clusters.run(OUTPUT_PATH_CLEAN, batch_to_be_combined = BTC, batch_length = MBS, output_path=OUTPUT_PATH_MERGE, llm = LLM_MERGE)
print("----------------FINE MERGE_CLUSTERS---------------")
fine = time.perf_counter()
print(f"Tempo di esecuzione totale: {(fine - inizio)/60:.2f} minuti")
