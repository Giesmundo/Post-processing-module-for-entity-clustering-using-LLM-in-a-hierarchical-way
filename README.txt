Documentazione Clean_Clusters.py

logica:

Far fare al LLM il minimo indispensabile, facendoli aggiornare dei valori booleani a seconda che una menzione faccia parte del cluster o meno, e creare cluster nuovi con menzioni orfane da se.
Creare un dict[doc_id, clusters_list] per ogni documento, partendo dalle clusters_list creare un
dict[doc_id, dict[tuple[mention_id, titolo_cluster, testo_menzione, contesto_menzione], bool]]
così che l’LLM possa aggiornare il valore booleano a TRUE se il titolo del cluster e il testo della menzione fanno riferimento alla stessa cosa, a FALSE altrimenti.
Partendo dal dict[doc_id, clusters_list] iniziale, verificare che le menzioni dei cluster dei documenti siano corretti comparando mention_id del dict ritornato dal LLM e quello del dict non modificato, creare nuovi cluster partendo dalle menzioni orfane (per futuro merge), ritornare il documento pulito

Implementazione:

separate_doc:

partendo dalla lista presa dal file in I, separa i cluster per ogni documento creando un 	dict[doc_id, clusters_list]

separate_mentions:

partendo dalle singole clusters_list, separa le menzioni creando un dict con solo le informazioni necessarie per verificare se una menzione faccia parte di un cluster o meno.
Ritorna un dict[tuple[mention_id, titolo_cluster, testo_menzione, contesto_menzione],bool] che viene poi messo come valore a dict[doc_id, …

process_mentions_parallel:

partendo dal dict di menzioni, per ogni documento prende le menzioni, le divide in chunk di grandezza prefissata con controllo di resti ed 	effettua chiamate parallele (4 per volta) al LLM con in I i chunk di menzioni tenendo traccia dell’ordine, aggiornando e ritornando il 	dict delle menzioni con quelle revisionate dal LLM

call_llm:

partendo dal chunk di menzioni, lo serializza e lo unisce al prompt, prova 	a chiamare l’LLM e pulisce il risultato creando un dict[tuple[mention_id, titolo_cluster, testo_menzione], bool] che ritorna

clean_clusters:

partendo dal dict[doc_id, clusters_list] di separate_doc e dal
dict[doc_id, dict[tuple[mention_id, titolo_cluster, testo_menzione],bool]] di process_mentions_parallel passa ogni menzione di ogni cluster di ogni documento e controlla, confrontyando le menzioni con il mention_id, se riferiscano al giusto elemento, 	altrimenti crea un cluster nuovo solo per quella menzione
