Documentazione Clean_Clusters.py

logica:

Far fare al LLM il minimo indispensabile, facendoli aggiornare dei valori booleani a seconda che una menzione faccia parte del cluster o meno, e creare cluster nuovi con menzioni orfane da se.
Creare un dict[doc_id, clusters_list] per ogni documento, partendo dalle clusters_list creare un
dict[doc_id, dict[mention_id, [cluster_title, mention_text, mention_context]]]
passare al LLM il dict[int, [cluster_title, mention_text, mention_context]] e fargli creare e ritornare un dict[int, bool] dove il bool può essere true, false, none a seconda
che, rispettivamente, la menzione: riferisce alla stessa entità del cluster, non riferisce alla stessa entità del cluster, non dovrebbe essere una menzione.

Implementazione:

separate_doc:

partendo dalla lista presa dal file in I, separa i cluster per ogni documento creando un dict[doc_id, clusters_list].

separate_mentions:

partendo dalle singole clusters_list, separa le menzioni creando un dict con solo le informazioni necessarie per verificare se una menzione faccia parte di un cluster o meno.
Ritorna un dict[mention_id, [cluster_title, mention_text, mention_context]] che viene poi messo come valore a dict[doc_id, …

process_mentions_parallel:

partendo dal dict di menzioni, per ogni documento prende le menzioni, le divide in chunk di grandezza prefissata con controllo di resti ed 	effettua chiamate parallele (8 per volta) al LLM con in I i chunk di menzioni tenendo traccia dell’ordine, aggiornando e ritornando il dict[doc_id, dict[mention_id, bool]] delle menzioni con quelle revisionate dal LLM.

call_llm:

partendo dal chunk di menzioni, lo serializza e lo unisce al prompt, prova a chiamare l’LLM e pulisce il risultato creando un dict[mention_id, bool] che ritorna.

clean_clusters:

partendo dal dict[doc_id, clusters_list] di separate_doc e daldict[doc_id, dict[mention_id, bool]] di process_mentions_parallel passa ogni menzione di ogni cluster di ogni documento e controlla 
confrontando le menzioni con il mention_id, se riferiscano al giusto elemento,se hanno valore Null le scarta, se hanno valore False crea un cluster nuovo solo per loro(aspettando merge_clusters)
se hanno valore true le lascia al loro posto.
