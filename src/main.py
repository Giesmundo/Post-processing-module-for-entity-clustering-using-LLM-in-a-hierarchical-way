import Clean_Clusters as Clean_Clusters
import Merge_Clusters as Merge_Clusters
import time

inizio = time.perf_counter()
with open(r"D:\Tesi PY\errors\errori.txt", "w") as f:
    f.write("")
print("----------------Inizio main-----------------------")
print("----------------INIZIO CLEAN_CLUSTERS-----------------")
O_path: str = Clean_Clusters.run(r"D:\Tesi PY\data\Cluster_Da_Pullire.json", context = True, NER_type = True, batch_length = 10)
print("----------------FINITO CLEAN_CLUSTERS-------------")
print("----------------INIZIO MERGE_CLUSTERS-------------")
O_path = Merge_Clusters.run(O_path, batch_to_be_combined = 2, batch_length = 50)
print("----------------FINE MERGE_CLUSTERS---------------")
fine = time.perf_counter()
print(f"Tempo di esecuzione totale: {(fine - inizio)/60:.2f} minuti")
