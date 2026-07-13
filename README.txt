Documentazione Clean_Clusters.py e Merge_Clusters.py

logica CC:
mandare al LLM i json di cluster con menzioni di singoli documenti di un singolo tipo, ritorna un dict in cui vengono segnati gli id dell emznioni da eliminare o da spostare
eseguire l'eliminazione e lo spostamento manualmente

logica MC:
mandare al LLM i titoli con id dei cluster di singoli documenti di un singolo tipo, ritorna un dict in cui vengono raggruppati gli id di cluster che si riferiscono alla stessa entità
successivamente prendere i titoli e gli id dei cluster di un singolo tipo di tutti i doucmenti, mandarli al LLM e nel caso siano troppo grandi suddividerli e usare una struttura ad albero per raggruppare i gruppi, una volta processati, mischiandoli per evitare la perdita di focus del LLM.
eseguire l'unione dei cluster manualmente.
