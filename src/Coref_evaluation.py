import json
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# 1. METRICHE DI BASE (MUC, B-cubed, CEAFe)
# ---------------------------------------------------------------------------

def muc(key_clusters, response_map):
    """
    MUC (Vilain et al., 1995).
    key_clusters: lista di cluster gold, ciascuno una lista/set di mention_id
    response_map: dict mention_id -> id del cluster di sistema a cui appartiene
                   (o None/assente se la menzione non e' nel sistema)
    Ritorna (numeratore, denominatore) per una singola direzione (recall se
    key_clusters = gold e response_map = mappa di sistema; precision se si
    invertono i ruoli).
    """
    num, den = 0, 0
    for c in key_clusters:
        c = list(c)
        den += len(c) - 1
        linked = set()
        for m in c:
            if m in response_map:
                linked.add(response_map[m])
        num += len(c) - len(linked)
    return num, den


def b_cubed(key_clusters, response_map):
    """
    B-cubed (Bagga & Baldwin, 1998).
    Stessa convenzione di input di muc().
    """
    num, den = 0.0, 0
    for c in key_clusters:
        c = list(c)
        if len(c) == 0:
            continue
        gold_counts = Counter()
        for m in c:
            if m in response_map:
                gold_counts[response_map[m]] += 1
        correct = sum(count * count for count in gold_counts.values())
        num += correct / float(len(c))
        den += len(c)
    return num, den


def _phi4(c1, c2):
    """Similarita' tra due cluster per CEAFe (Luo et al., 2005)."""
    c1, c2 = set(c1), set(c2)
    return 2 * len(c1 & c2) / float(len(c1) + len(c2))


def ceafe(key_clusters, response_clusters):
    """
    CEAFe (entity-based CEAF, Luo et al., 2005).
    Ritorna (similarity, |response_clusters|, similarity, |key_clusters|),
    cioe' (num_precision, den_precision, num_recall, den_recall).
    """
    key_clusters = [c for c in key_clusters if len(c) > 0]
    response_clusters = [c for c in response_clusters if len(c) > 0]

    scores = np.zeros((len(key_clusters), len(response_clusters)))
    for i, kc in enumerate(key_clusters):
        for j, rc in enumerate(response_clusters):
            scores[i, j] = _phi4(kc, rc)

    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()

    return similarity, len(response_clusters), similarity, len(key_clusters)


# ---------------------------------------------------------------------------
# 2. AGGREGATORE (accumula su piu' documenti, poi calcola F1 alla fine --
#    stessa convenzione del CoNLL scorer ufficiale)
# ---------------------------------------------------------------------------

class CorefEvaluator:
    def __init__(self):
        self.p_num, self.p_den = 0, 0
        self.r_num, self.r_den = 0, 0
        # per CEAFe teniamo somme separate perche' la formula e' diversa
        self.ceafe_p_num, self.ceafe_p_den = 0, 0
        self.ceafe_r_num, self.ceafe_r_den = 0, 0
        self.metric = "conll"  # etichetta informativa

    def update_muc_b3(self, key_clusters, response_clusters,
                       key_map, response_map, metric_fn):
        r_num, r_den = metric_fn(key_clusters, response_map)
        p_num, p_den = metric_fn(response_clusters, key_map)
        self.p_num += p_num
        self.p_den += p_den
        self.r_num += r_num
        self.r_den += r_den

    def update_ceafe(self, key_clusters, response_clusters):
        p_num, p_den, r_num, r_den = ceafe(key_clusters, response_clusters)
        self.ceafe_p_num += p_num
        self.ceafe_p_den += p_den
        self.ceafe_r_num += r_num
        self.ceafe_r_den += r_den

    @staticmethod
    def _prf(p_num, p_den, r_num, r_den):
        p = p_num / p_den if p_den > 0 else 0.0
        r = r_num / r_den if r_den > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1


def build_mention_map(clusters):
    """
    clusters: lista di cluster, ciascuno una lista di mention_id (es. tuple (start,end))
    Ritorna un dict mention_id -> indice del cluster a cui appartiene.
    """
    m = {}
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            m[mention] = cluster_id
    return m


def evaluate_document(gold_clusters, sys_clusters):
    """
    gold_clusters, sys_clusters: liste di cluster (liste di mention_id) per UN documento.
    Ritorna un dict con MUC, B3, CEAFe (num/den) per questo documento, da accumulare.
    """
    gold_map = build_mention_map(gold_clusters)
    sys_map = build_mention_map(sys_clusters)

    muc_r = muc(gold_clusters, sys_map)
    muc_p = muc(sys_clusters, gold_map)

    b3_r = b_cubed(gold_clusters, sys_map)
    b3_p = b_cubed(sys_clusters, gold_map)

    ceafe_res = ceafe(gold_clusters, sys_clusters)  # (p_num, p_den, r_num, r_den)

    return {
        "muc": {"p": muc_p, "r": muc_r},
        "b3": {"p": b3_p, "r": b3_r},
        "ceafe": ceafe_res,
    }


def evaluate_corpus(gold_by_doc, sys_by_doc):
    """
    gold_by_doc, sys_by_doc: dict {doc_id: [cluster1, cluster2, ...]}
    dove ogni cluster e' una lista di mention_id (es. tuple (start, end)).
    Aggrega su tutti i documenti (stile CoNLL scorer ufficiale) e calcola
    precision/recall/F1 per MUC, B3, CEAFe, oltre al CoNLL-F1 medio.

    ATTENZIONE (bug noto, mantenuto qui solo per riferimento storico): questa
    funzione accumula in un unico accumulatore (ev.p_num/p_den/...) sia MUC
    che B3, mischiando le due metriche. Non usarla: usa
    evaluate_corpus_clean() qui sotto, che tiene accumulatori separati.
    """
    ev = CorefEvaluator()

    for doc_id in gold_by_doc:
        gold_clusters = gold_by_doc[doc_id]
        sys_clusters = sys_by_doc.get(doc_id, [])

        gold_map = build_mention_map(gold_clusters)
        sys_map = build_mention_map(sys_clusters)

        # MUC
        ev.update_muc_b3(gold_clusters, sys_clusters, gold_map, sys_map, muc)
        # B3
        ev.update_muc_b3(gold_clusters, sys_clusters, gold_map, sys_map, b_cubed)
        # CEAFe (gestito separatamente perche' non usa la stessa firma)
        ev.update_ceafe(gold_clusters, sys_clusters)

    # NB: p_num/p_den/r_num/r_den di CorefEvaluator qui contengono la somma
    # cumulata di MUC + B3 insieme, che NON va bene: li separiamo qui sotto
    # ricalcolando in modo pulito senza mischiare le metriche.
    return ev


# La versione sopra mischierebbe MUC e B3 nello stesso accumulatore: qui sotto
# la versione corretta e pronta all'uso, con accumulatori separati per metrica.

def evaluate_corpus_clean(gold_by_doc, sys_by_doc):
    """
    Versione corretta (accumulatori separati per MUC/B3/CEAFe).

    NOTA (coreference evaluation, non mention detection): questa funzione
    presuppone che gold_by_doc e sys_by_doc contengano gia' SOLO le mention
    su cui le due parti "concordano" nell'esistere (tipicamente l'insieme
    intersezione gold ∩ sistema -- vedi filter_to_common_mentions()). MUC,
    B3 e CEAFe misurano infatti la qualita' delle DECISIONI DI LINKING
    (quali mention sono state raggruppate correttamente insieme), non la
    qualita' del mention detection (quali mention sono state trovate). Se in
    gold_by_doc/sys_by_doc restano mention presenti solo da un lato, queste
    tre metriche le trattano semplicemente come "non linkate", il che non e'
    la stessa cosa di penalizzare esplicitamente un errore di detection.
    """
    muc_p_num = muc_p_den = muc_r_num = muc_r_den = 0
    b3_p_num = b3_p_den = b3_r_num = b3_r_den = 0.0
    ceafe_p_num = ceafe_p_den = ceafe_r_num = ceafe_r_den = 0.0

    for doc_id in gold_by_doc:
        gold_clusters = gold_by_doc[doc_id]
        sys_clusters = sys_by_doc.get(doc_id, [])

        gold_map = build_mention_map(gold_clusters)
        sys_map = build_mention_map(sys_clusters)

        # MUC
        r_num, r_den = muc(gold_clusters, sys_map)
        p_num, p_den = muc(sys_clusters, gold_map)
        muc_r_num += r_num; muc_r_den += r_den
        muc_p_num += p_num; muc_p_den += p_den

        # B3
        r_num, r_den = b_cubed(gold_clusters, sys_map)
        p_num, p_den = b_cubed(sys_clusters, gold_map)
        b3_r_num += r_num; b3_r_den += r_den
        b3_p_num += p_num; b3_p_den += p_den

        # CEAFe
        p_num, p_den, r_num, r_den = ceafe(gold_clusters, sys_clusters)
        ceafe_p_num += p_num; ceafe_p_den += p_den
        ceafe_r_num += r_num; ceafe_r_den += r_den

    def prf(p_num, p_den, r_num, r_den):
        p = p_num / p_den if p_den > 0 else 0.0
        r = r_num / r_den if r_den > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    muc_p, muc_r, muc_f1 = prf(muc_p_num, muc_p_den, muc_r_num, muc_r_den)
    b3_p, b3_r, b3_f1 = prf(b3_p_num, b3_p_den, b3_r_num, b3_r_den)
    ceafe_p, ceafe_r, ceafe_f1 = prf(ceafe_p_num, ceafe_p_den, ceafe_r_num, ceafe_r_den)

    conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3.0

    return {
        "muc": {"precision": muc_p, "recall": muc_r, "f1": muc_f1},
        "b3": {"precision": b3_p, "recall": b3_r, "f1": b3_f1},
        "ceafe": {"precision": ceafe_p, "recall": ceafe_r, "f1": ceafe_f1},
        "conll_f1": conll_f1,
    }


# ---------------------------------------------------------------------------
# 2bis. METRICHE "STILE ER" (ACC, FP-measure, NMI, ARI) -- famiglia usata da
#       LLM-CER e da gran parte della letteratura di Entity Resolution/clustering
#       generico. Calcolate sullo STESSO input (cluster gold + cluster di sistema)
#       usato per MUC/B3/CEAFe, cosi' puoi riportare entrambe le famiglie di
#       metriche fianco a fianco per lo stesso output del tuo modello.
#
# DIFFERENZA CHIAVE rispetto a MUC/B3/CEAFe: qui NON si filtra sulle mention
# comuni. ACC, purity/inverse-purity/FP-measure, NMI e ARI sono definite
# sull'UNIVERSO gold ∪ sistema (vedi _mention_universe()), quindi una mention
# che il sistema ha "inventato" (assente in gold) o che ha perso (presente
# solo in gold) resta nell'universo e pesa nel denominatore senza poter
# contribuire al numeratore/all'overlap corretto: in questo modo queste
# metriche catturano insieme sia la qualita' del linking sia quella del
# mention detection, a differenza di MUC/B3/CEAFe che misurano SOLO il
# linking sulle mention condivise.
# ---------------------------------------------------------------------------

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def _mention_universe(gold_clusters, sys_clusters):
    """Unione di tutte le menzioni presenti nel gold e/o nel sistema."""
    universe = set()
    for c in gold_clusters:
        universe.update(c)
    for c in sys_clusters:
        universe.update(c)
    return sorted(universe)


def _labels_from_clusters(clusters, universe):
    """
    Converte una lista di cluster in un array di etichette (una per menzione
    dell'universo, nello stesso ordine). Le menzioni non presenti in nessun
    cluster ricevono un'etichetta singleton univoca (cosi' non "matchano" per
    caso nessun altro cluster).
    """
    mention_to_label = {}
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_label[mention] = cluster_id

    labels = []
    next_singleton_id = len(clusters)
    for mention in universe:
        if mention in mention_to_label:
            labels.append(mention_to_label[mention])
        else:
            labels.append(next_singleton_id)
            next_singleton_id += 1
    return labels


def accuracy(gold_clusters, sys_clusters, universe):
    """
    ACC come definita in LLM-CER (Fu et al. 2025, Eq. 2-3): allinea in modo
    ottimale i cluster di sistema con i cluster gold (massimizzando l'overlap
    totale, tramite l'algoritmo ungherese) e conta quante menzioni finiscono
    nel cluster gold corrispondente.

    Lavora sull'universo gold ∪ sistema: il denominatore e' len(universe), che
    include anche le mention extra del sistema e quelle mancanti rispetto al
    gold. Queste ultime non possono mai contribuire al numeratore (overlap),
    quindi vengono penalizzate automaticamente -- a differenza di MUC/B3/CEAFe,
    che sono calcolate solo sulle mention condivise.
    """
    if not gold_clusters or not sys_clusters:
        return 0.0

    n_gold, n_sys = len(gold_clusters), len(sys_clusters)
    size = max(n_gold, n_sys)
    overlap = np.zeros((size, size))
    for i in range(n_gold):
        gi = set(gold_clusters[i])
        for j in range(n_sys):
            overlap[i, j] = len(gi & set(sys_clusters[j]))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    correct = overlap[row_ind, col_ind].sum()

    return correct / len(universe) if universe else 0.0


def purity_inverse_purity_fp(gold_clusters, sys_clusters, universe):
    """
    Purity, inverse-purity e FP-measure come definite in LLM-CER (Eq. 4-7).
    X = cluster di sistema (predicted), Y = cluster gold (ground truth).

    Anche qui N = len(universe) = |gold ∪ sistema|, quindi mention extra o
    mancanti riducono comunque purity/inverse-purity (pesano nel denominatore
    ma non possono contribuire all'overlap "corretto" nel numeratore).
    """
    N = len(universe)
    if N == 0:
        return 0.0, 0.0, 0.0

    X = [set(c) for c in sys_clusters]
    Y = [set(c) for c in gold_clusters]

    # purity: quanto ogni cluster di sistema e' "puro" rispetto al gold migliore
    purity = 0.0
    for Xi in X:
        if not Xi:
            continue
        best_overlap = max((len(Xi & Yj) for Yj in Y), default=0)
        purity += best_overlap  # gia' pesato correttamente sommando i conteggi grezzi
    purity /= N

    # inverse-purity: stessa idea ma con i ruoli invertiti (X <-> Y)
    inverse_purity = 0.0
    for Yj in Y:
        if not Yj:
            continue
        best_overlap = max((len(Yj & Xi) for Xi in X), default=0)
        inverse_purity += best_overlap
    inverse_purity /= N

    fp_measure = (2 / (1 / purity + 1 / inverse_purity)
                  if purity > 0 and inverse_purity > 0 else 0.0)

    return purity, inverse_purity, fp_measure


def evaluate_er_metrics(gold_clusters, sys_clusters):
    """
    Calcola ACC, purity, inverse-purity, FP-measure, NMI, ARI per UN documento
    (o per l'intera collezione, se gli passi tutti i cluster concatenati).

    Tutte queste metriche sono definite sull'universo gold ∪ sistema (vedi
    _mention_universe): a differenza di MUC/B3/CEAFe (che misurano solo la
    qualita' del linking sulle mention che entrambe le parti riconoscono),
    ACC/purity/inverse-purity/FP-measure/NMI/ARI incorporano anche l'effetto
    di eventuali errori di mention detection (mention extra del sistema o
    mention del gold non trovate), perche' quelle mention restano
    nell'universo come singoletti "orfani" e pesano nel calcolo.
    """
    universe = _mention_universe(gold_clusters, sys_clusters)

    acc = accuracy(gold_clusters, sys_clusters, universe)
    purity, inv_purity, fp = purity_inverse_purity_fp(gold_clusters, sys_clusters, universe)

    labels_gold = _labels_from_clusters(gold_clusters, universe)
    labels_sys = _labels_from_clusters(sys_clusters, universe)

    nmi = normalized_mutual_info_score(labels_gold, labels_sys)
    ari = adjusted_rand_score(labels_gold, labels_sys)

    return {
        "acc": acc,
        "purity": purity,
        "inverse_purity": inv_purity,
        "fp_measure": fp,
        "nmi": nmi,
        "ari": ari,
    }


def evaluate_corpus_er_metrics(gold_by_doc, sys_by_doc):
    """
    Versione a livello di intera collezione (concatena i cluster di tutti i
    documenti, dato che le metriche ER/LLM-CER non sono definite per singolo
    documento nello stesso senso "micro-average" del CoNLL scorer).
    """
    all_gold_clusters = []
    all_sys_clusters = []
    for doc_id in gold_by_doc:
        # NB: per evitare falsi overlap tra menzioni di documenti diversi che
        # avessero per assurdo lo stesso (start, end), qui rendiamo l'id di
        # menzione univoco a livello di collezione includendo il doc_id.
        all_gold_clusters.extend(
            [[(doc_id, m) for m in c] for c in gold_by_doc[doc_id]]
        )
        all_sys_clusters.extend(
            [[(doc_id, m) for m in c] for c in sys_by_doc.get(doc_id, [])]
        )
    return evaluate_er_metrics(all_gold_clusters, all_sys_clusters)


# ---------------------------------------------------------------------------
# 3. CONVERSIONE DAL TUO FORMATO JSON
# ---------------------------------------------------------------------------
#
# NOTA IMPORTANTE SULL'IDENTIFICAZIONE DELLE MENZIONI
# ----------------------------------------------------
# Il campo "id" delle menzioni e' univoco solo all'INTERNO di un singolo
# documento originale, non a livello dell'intera collezione. Fondendo piu'
# documenti (come accade qui), lo stesso numero di "id" puo' comparire piu'
# volte riferendosi a menzioni diverse (testo/posizione diversi).
#
# Per questo motivo NON usiamo "id" da solo come identificatore di menzione,
# ma una CHIAVE COMPOSITA (id, start, end, text): la probabilita' che due
# menzioni realmente distinte abbiano contemporaneamente stesso id, stesso
# start, stesso end e stesso testo e' trascurabile, quindi questa chiave
# funge da identificatore univoco affidabile senza richiedere un originalDocId
# esplicito (che, tra l'altro, puo' mancare nei cluster gia' fusi cross-document,
# come nel tuo formato di output "merge_extra_document_...").
#
# Di conseguenza, l'intera collezione viene trattata come UN SOLO "documento"
# ai fini della valutazione (non si raggruppa piu' per originalDocId): questo
# e' corretto e anzi necessario, perche' un cluster gia' fuso cross-document
# non appartiene comunque a un singolo documento.


def _mention_key(mention):
    """Costruisce l'identificatore univoco di una menzione (vedi nota sopra)."""
    return (mention["id"], mention["start"], mention["end"], mention["text"])


def gt_json_to_clusters(gt_path):
    """
    Converte il formato della TUA GROUND TRUTH:
        { "nome entita' 1": [ {mention...}, {mention...} ],
          "nome entita' 2": [ {mention...} ], ... }
    in una lista di cluster, ciascuno una lista di chiavi di menzione.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    clusters = []
    for entity_name, mentions in gt_raw.items():
        cluster = [_mention_key(m) for m in mentions if m is not None]
        if cluster:
            clusters.append(cluster)
    return clusters


def system_json_to_clusters(system_path):
    """
    Converte il formato di OUTPUT DEL TUO MODELLO:
        [ {"clusterId": ..., "mentions": [ {mention...}, ... ], ...},
          {"clusterId": ..., "mentions": [ ... ], ...}, ... ]
    in una lista di cluster, ciascuno una lista di chiavi di menzione.

    Nota: "originalDocId" viene ignorato di proposito (vedi nota sopra: puo'
    mancare nei cluster gia' fusi cross-document, e comunque non serve piu'
    dato che valutiamo l'intera collezione come un unico spazio).
    """
    with open(system_path, "r", encoding="utf-8") as f:
        system_raw = json.load(f)

    clusters = []
    for cluster_obj in system_raw:
        if cluster_obj is None:
            continue
        mentions = cluster_obj.get("mentions", [])
        cluster = [_mention_key(m) for m in mentions if m is not None]
        if cluster:
            clusters.append(cluster)
    return clusters


# Manteniamo anche la vecchia funzione per compatibilita', ma non e' piu'
# quella da usare con i tuoi due formati reali (GT a dizionario, output con
# cluster eventualmente privi di originalDocId). Usa gt_json_to_clusters()
# e system_json_to_clusters() sopra.
def json_to_clusters_by_doc(json_path, doc_id_field="originalDocId"):
    """
    Converte il tuo formato JSON (lista di cluster con 'mentions' contenenti
    'start' ed 'end') in un dict {doc_id: [cluster1, cluster2, ...]},
    dove ogni cluster e' una lista di mention_id = (start, end).

    ATTENZIONE: gli offset (start, end) devono riferirsi allo stesso sistema
    di coordinate sia nel file gold sia nel file di sistema (stesso documento
    originale, stessa tokenizzazione/offset), altrimenti il confronto non ha senso.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        clusters_raw = json.load(f)

    by_doc = defaultdict(list)
    for cluster in clusters_raw:
        if cluster is None:
            continue
        doc_id = cluster[doc_id_field]
        mention_ids = []
        for mention in cluster.get("mentions", []):
            if mention is None:
                continue
            mention_id = (mention["start"], mention["end"])
            mention_ids.append(mention_id)
        if mention_ids:
            by_doc[doc_id].append(mention_ids)

    return dict(by_doc)


# ---------------------------------------------------------------------------
# 4. GESTIONE MENTION DISALLINEATE TRA GOLD E SISTEMA
# ---------------------------------------------------------------------------

def filter_to_common_mentions(gold_clusters, sys_clusters):
    """
    Restringe gold_clusters e sys_clusters alle sole mention presenti SIA nel
    gold SIA nel sistema (intersezione), senza inventare nulla.

    Questa e' la funzione da usare prima di calcolare MUC/B3/CEAFe: queste
    tre metriche misurano la qualita' delle decisioni di CLUSTERING/LINKING
    (se due mention che esistono in entrambe le annotazioni sono state messe
    correttamente nello stesso cluster o meno). Non sono pensate per valutare
    il MENTION DETECTION, cioe' se il sistema ha trovato le mention giuste:
    quella e' una domanda diversa, gia' misurata a parte (per esempio con
    precision/recall sulle mention stesse, oppure in modo implicito da
    ACC/NMI/ARI che lavorano sull'universo gold ∪ sistema, vedi sotto).

    A differenza della vecchia pad_clusters_for_mismatched_mentions() (RIMOSSA),
    qui NON si aggiunge alcun cluster singoletto fittizio per le mention
    disallineate: quelle mention vengono semplicemente escluse dal calcolo di
    MUC/B3/CEAFe. I cluster che restano vuoti dopo il filtro vengono scartati.

    Ritorna (filtered_gold_clusters, filtered_sys_clusters).
    """
    gold_mentions = set(m for c in gold_clusters for m in c)
    sys_mentions = set(m for c in sys_clusters for m in c)
    common = gold_mentions & sys_mentions

    filtered_gold = [[m for m in c if m in common] for c in gold_clusters]
    filtered_gold = [c for c in filtered_gold if c]

    filtered_sys = [[m for m in c if m in common] for c in sys_clusters]
    filtered_sys = [c for c in filtered_sys if c]

    return filtered_gold, filtered_sys


def evaluate_collection(gold_clusters, sys_clusters):
    """
    Versione "a collezione unica": prende direttamente le due liste piatte di
    cluster (gold e sistema) sull'intera collezione e calcola MUC/B3/CEAFe/
    CoNLL-F1.

    Applica filter_to_common_mentions() prima del calcolo: MUC/B3/CEAFe
    vengono quindi calcolate SOLO sulle mention presenti sia in gold sia nel
    sistema (coreference evaluation in senso stretto). Le mention che il
    sistema ha inventato (assenti in gold) o che ha perso (presenti solo in
    gold) non vengono trasformate in singoletti artificiali e non entrano nel
    calcolo di queste tre metriche: se vuoi che l'errore di mention detection
    si rifletta nel punteggio, guarda invece evaluate_collection_er_metrics()
    (ACC/purity/inverse-purity/FP-measure/NMI/ARI), che lavora sull'universo
    gold ∪ sistema.
    """
    common_gold, common_sys = filter_to_common_mentions(gold_clusters, sys_clusters)
    return evaluate_corpus_clean(
        gold_by_doc={"collection": common_gold},
        sys_by_doc={"collection": common_sys},
    )


def evaluate_collection_er_metrics(gold_clusters, sys_clusters):
    """
    Equivalente 'a collezione unica' per le metriche ER (ACC/purity/
    inverse-purity/FP-measure/NMI/ARI).

    A differenza di evaluate_collection() (MUC/B3/CEAFe), qui si usano
    gold_clusters e sys_clusters COSI' COME SONO, senza alcun filtro sulle
    mention comuni: l'universo di riferimento e' gold ∪ sistema (vedi
    _mention_universe/evaluate_er_metrics), quindi mention extra o mancanti
    pesano nel punteggio finale. Questa e' la ragione per cui, a parita' di
    output del modello, ACC/NMI/ARI possono risultare piu' bassi di quanto
    ci si aspetterebbe guardando solo MUC/B3/CEAFe: le prime penalizzano
    anche gli errori di mention detection, le seconde no.
    """
    return evaluate_er_metrics(gold_clusters, sys_clusters)


# ---------------------------------------------------------------------------
# 5. ESEMPIO D'USO (con i TUOI formati reali: GT a dizionario, output a lista
#    di cluster con originalDocId opzionale)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GOLD_PATH = r"D:\Tesi PY\Ground_truth\th12_Gold_Clusters.json"#r"D:\Tesi PY\Ground_truth\M_GROUND_TRUTH.json"        # il tuo file GT: { "nome entita'": [mentions] }
    SYSTEM_PATH = r"D:\Tesi PY\output\th12_Clusters_Uniti.json"#r"D:\LLM_CER\llmcer_output_converted.json"  # il tuo file di output del modello

    gold_clusters = gt_json_to_clusters(GOLD_PATH)
    sys_clusters = system_json_to_clusters(SYSTEM_PATH)

    print(f"Cluster gold: {len(gold_clusters)}  |  Cluster di sistema: {len(sys_clusters)}")

    # MUC / B3 / CEAFe: solo sulle mention condivise tra gold e sistema
    # (coreference evaluation in senso stretto, vedi filter_to_common_mentions).
    results = evaluate_collection(gold_clusters, sys_clusters)

    print("\n=== Famiglia CoNLL / CDCR (compatibile con xCoRe, Cattan et al., CDLM) ===")
    print("Calcolate SOLO sulle mention presenti sia in gold sia nel sistema.")
    print("MUC   : P={:.4f}  R={:.4f}  F1={:.4f}".format(
        results["muc"]["precision"], results["muc"]["recall"], results["muc"]["f1"]))
    print("B3    : P={:.4f}  R={:.4f}  F1={:.4f}".format(
        results["b3"]["precision"], results["b3"]["recall"], results["b3"]["f1"]))
    print("CEAFe : P={:.4f}  R={:.4f}  F1={:.4f}".format(
        results["ceafe"]["precision"], results["ceafe"]["recall"], results["ceafe"]["f1"]))
    print("CoNLL-F1 (media di MUC/B3/CEAFe): {:.4f}".format(results["conll_f1"]))

    # ACC / purity / inverse-purity / FP-measure / NMI / ARI: sull'universo
    # gold ∪ sistema, quindi includono anche l'effetto di mention extra/mancanti.
    er_results = evaluate_collection_er_metrics(gold_clusters, sys_clusters)
    print("\n=== Famiglia ER / clustering generico (compatibile con LLM-CER) ===")
    print("Calcolate su gold ∪ sistema: penalizzano anche mention extra/mancanti.")
    print("ACC            : {:.4f}".format(er_results["acc"]))
    print("Purity         : {:.4f}".format(er_results["purity"]))
    print("Inverse-Purity : {:.4f}".format(er_results["inverse_purity"]))
    print("FP-measure     : {:.4f}".format(er_results["fp_measure"]))
    print("NMI            : {:.4f}".format(er_results["nmi"]))
    print("ARI            : {:.4f}".format(er_results["ari"]))