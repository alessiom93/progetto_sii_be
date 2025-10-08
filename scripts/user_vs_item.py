import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from item_based_cf_rs import item_based_cf
from user_based_cf_rs import user_based_cf

''' Confronto tra User-Based e Item-Based Collaborative Filtering
1. Campionamento denso: estrae un sottoinsieme del dataset con utenti e item con più valutazioni.
2. Selezione utenti: sceglie un sottoinsieme casuale di utenti per il test.
3. Creazione test set: per ogni utente selezionato, estrae casualmente il 20% delle sue valutazioni come test set.
4. Raccomandazioni: per ogni utente nel test set, calcola raccomandazioni usando entrambi gli algoritmi
5. Valutazione: calcola hit rate, precision e recall confrontando le raccomandazioni con il test set.
6. Output: restituisce le metriche di valutazione per entrambi gli algoritmi.
'''

logger = logging.getLogger(__name__)

# Fa un campionamento denso del dataset
# per testare gli algoritmi su un sottoinsieme più gestibile.
def sample_dense_subset(ratings_df, n_users=20, n_items=20, n_ratings=200, random_state=42):
    print(f"[DEBUG] Set originale: {ratings_df.shape} shape, {ratings_df['User-ID'].nunique()} utenti, {ratings_df['ISBN'].nunique()} item")
    # seed per riproducibilità
    np.random.seed(random_state)
    # seleziona gli n utenti con più valutazioni
    top_users = ratings_df["User-ID"].value_counts().head(n_users).index
    # seleziona gli n item con più valutazioni
    top_items = ratings_df["ISBN"].value_counts().head(n_items).index
    # filtra il dataframe originale per ottenere il sottoinsieme denso
    # include solo le valutazioni degli utenti e item selezionati
    subset = ratings_df[
        ratings_df["User-ID"].isin(top_users) & ratings_df["ISBN"].isin(top_items)
    ]
    # calcola il numero massimo possibile di valutazioni in questo sottoinsieme
    max_possible = n_users * n_items
    # limita il numero di valutazioni a n_ratings o al massimo possibile
    n_ratings = min(n_ratings, max_possible, len(subset))
    # estrae un sottoinsieme casuale di n_ratings dal dataframe filtrato
    subset = subset.sample(n=n_ratings, random_state=random_state)
    # calcola la densità del sottoinsieme (percentuale di celle non nulle nella matrice user-item)
    density = n_ratings / max_possible if max_possible > 0 else 0
    logger.debug(f"[DEBUG] Subset campionato: {subset.shape[0]} righe, {subset['User-ID'].nunique()} utenti, {subset['ISBN'].nunique()} item, densità={density:.2f}")
    print(f"[DEBUG] Subset campionato: {subset.shape} shape, {subset['User-ID'].nunique()} utenti, {subset['ISBN'].nunique()} item, densità={density:.2f}")
    # ritorna il sottoinsieme resettando gli indici
    return subset.reset_index(drop=True)

# Valuta le raccomandazioni calcolando hit rate, precision e recall
# preds_dict: dizionario {user_id: [(ISBN, predicted_rating), ...]}
# test_user_items: dizionario {user_id: set(ISBN)}
# top_n: numero di raccomandazioni da considerare per utente
def evaluate_recommendations(preds_dict, test_user_items, top_n=10):
    # inizializza hit rate, precision e recall
    hit_count = 0
    precision_list, recall_list = [], []
    # cicla su ogni utente e le sue raccomandazioni
    for u, recs in preds_dict.items():
        # prendi solo i top_n libri raccomandati
        recs_top = [isbn for isbn, _ in recs[:top_n]]
        # recupera i libri effettivamente valutati dall'utente nel test set
        true_items = test_user_items.get(u, set())
        # calcola il numero di hit (libri raccomandati che sono nel test set)
        hits = len(set(recs_top) & true_items)
        hit_count += hits
        # calcola precision, ovvero la frazione di raccomandazioni corrette (recuperati & rilevanti / recuperati)
        precision_list.append(hits / top_n if top_n > 0 else 0)
        # calcola recall, ovvero la frazione di libri del test set che sono stati raccomandati (recuperati & rilevanti / rilevanti)
        recall_list.append(hits / len(true_items) if len(true_items) > 0 else 0)
    # calcola la hit rate complessiva, percentuale di item del test set raccomandati
    hit_rate = hit_count / sum(len(v) for v in test_user_items.values()) if test_user_items else 0
    # calcola precision e recall medi
    precision = np.nanmean(precision_list) if precision_list else 0
    recall = np.nanmean(recall_list) if recall_list else 0
    # ritorna le metriche
    return {"hit_rate": hit_rate, "precision": precision, "recall": recall}

# Parallelizzazione
def parallel_recommendations(func, users, train_df, k, top_n, min_common=1, max_workers=16):
    results = {}
    # crea un pool di processi con un massimo di max_workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # sottomette un task asincrono per ogni utente
        futures = {executor.submit(func, u, train_df, k, top_n, min_common): u for u in users}
        # attende che i task siano completati
        for future in as_completed(futures):
            u = futures[future]
            try:
                # salva i risultati
                results[str(u)] = future.result()
            except Exception as e:
                logger.warning(f"[ParallelProcess] Errore per utente {u}: {e}")
    # ritorna i risultati (dizionario {user_id: recommendations})
    return results

# Confronto algoritmi
def compare_user_item_cf(ratings_df, top_n=10, sample_n_users=500, k=50,
                         use_dense=True, n_users=500, n_items=500, n_ratings=150000,
                         use_parallel=True, max_workers=16):
    try:
        logger.info("Comparing user-based and item-based CF...")
        logger.info(f"Original ratings shape: {ratings_df.shape}")
        logger.info(f"Parameters: top_n={top_n}, sample_n_users={sample_n_users}, k={k}, use_dense={use_dense}, n_users={n_users}, n_items={n_items}, n_ratings={n_ratings}, use_parallel={use_parallel}, max_workers={max_workers}")
        # se richiesto, campiona un sotto-dataset denso
        if use_dense:
            ratings_df = sample_dense_subset(ratings_df, n_users=n_users, n_items=n_items, n_ratings=n_ratings)
        # estrae tutti gli ID unici degli utenti
        all_users = ratings_df['User-ID'].unique()
        # campiona casualmente un sottoinsieme di utenti per il test
        sampled_users = np.random.choice(all_users, size=min(sample_n_users, len(all_users)), replace=False)
        # dizionario per contenetere gli item di test per ogni utente
        test_user_items = {}
        for uid in sampled_users:
            # estrae le valutazioni dell'utente
            user_ratings = ratings_df[ratings_df['User-ID'] == uid]
            # se l'utente ha meno di 2 valutazioni, salta
            if len(user_ratings) < 2:
                continue
            # seleziona casualmente il 20% delle valutazioni come test set
            test_items = set(user_ratings.sample(frac=0.2, random_state=42)['ISBN'])
            # rimuove le valutazioni di test dal dataset di addestramento (non utilizzato)
            #train_ratings = ratings_df.drop(user_ratings[user_ratings['ISBN'].isin(test_items)].index)
            # registra gli item di test per l'utente
            test_user_items[str(uid)] = test_items
        # crea training set rimuovendo il test set
        print(f"Training ratings shape before removing test items: {ratings_df.shape}")
        train_ratings = ratings_df[~ratings_df.apply(
            lambda row: str(row['User-ID']) in test_user_items and row['ISBN'] in test_user_items[str(row['User-ID'])],
            axis=1
        )]
        print(f"Training ratings shape after removing test items: {train_ratings.shape}")
        # calcola le raccomandazioni usando entrambi gli algoritmi
        if use_parallel:
            # se richiesto, esecuzione parallela dei due algoritmi
            preds_user = parallel_recommendations(user_based_cf, sampled_users, train_ratings, k, top_n, min_common=1, max_workers=max_workers)
            preds_item = parallel_recommendations(item_based_cf, sampled_users, train_ratings, k, top_n, min_common=1, max_workers=max_workers)
        else:
            # altrimenti, esecuzione sequenziale
            preds_user = {str(u): user_based_cf(u, train_ratings, k=k, top_n=top_n) for u in sampled_users}
            preds_item = {str(u): item_based_cf(u, train_ratings, k=k, top_n=top_n) for u in sampled_users}
        # valuta le raccomandazioni calcolando hit rate, precision e recall
        user_metrics = evaluate_recommendations(preds_user, test_user_items, top_n=top_n)
        item_metrics = evaluate_recommendations(preds_item, test_user_items, top_n=top_n)
        # ritorna i risultati del confronto
        results = {"user_based": user_metrics, "item_based": item_metrics}
        logger.debug(f"Comparison results: {results}")
        # ritorna i risultati
        return results

    except Exception as e:
        logger.error(f"Error comparing user-based and item-based CF: {e}")
        return {}
