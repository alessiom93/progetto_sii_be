import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

''' User-Based Collaborative Filtering Recommender System 
1. Costruzione della matrice: crea una matrice users x items.
2. Normalizzazione: calcola la media dei rating per utente e centra i dati.
3. Calcolo similarità: trova utenti simili usando la cosine similarity.
4. Selezione vicini: mantiene solo i k utenti più simili che hanno abbastanza item in comune.
5. Predizione: usa una media pesata delle differenze dai rating medi dei vicini.
6. Output: restituisce i top_n libri con rating predetto più alto.
'''

# costruisce la matrice user-item
# input: dataframe con colonne User-ID, ISBN, Book-Rating
# output: matrice user-item + mappings tra indici e ID originali
def build_user_item_matrix(ratings: pd.DataFrame):
    # copia del dataframe su cui lavorare
    ratings = ratings.copy()
    # conversione degli ID in stringhe per consistenza
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    # conversione dei valori in categorie, ovvero in indici numerici, perchè più efficienti
    user_cat = ratings['User-ID'].astype('category')
    item_cat = ratings['ISBN'].astype('category')
    # mapping tra indici numerici e ID originali
    user_mapping = dict(enumerate(user_cat.cat.categories))   # index -> User-ID
    item_mapping = dict(enumerate(item_cat.cat.categories))   # index -> ISBN
    # mapping inversi da ID a indici numerici
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))  # User-ID -> index
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))  # ISBN -> index
    # costruzione matrice user-item, righe=users, colonne=items, valori=ratings
    # La matrice CSR è rappresentata da tre array: data (i ratings non nulli), indices (indici colonna della posizione dei ratings), indptr (indici di inizio/fine di ogni riga)
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float), # valori dei rating
         (user_cat.cat.codes, item_cat.cat.codes)), # coordinate (riga, colonna)
        shape=(len(user_mapping), len(item_mapping)) # dimensioni (n_users, n_items)
    )
    # ritorna matrice + mappings
    return mat, user_mapping, item_mapping, user_codes, item_codes

# user-based collaborative filtering
# input: user_id, dataframe con colonne User-ID, ISBN, Book-Rating
#        k=numero di users vicini, top_n=numero di raccomandazioni, min_common=minimo numero di item in comune tra utenti
#        min_rating, max_rating = range dei ratings
# output: lista di tuple (ISBN, predicted_rating) ordinate per predicted_rating decrescente
#         oppure lista vuota se non possibile fare raccomandazioni
def user_based_cf(
    user_id, ratings: pd.DataFrame, 
    k=50, 
    top_n=10, 
    min_common=1, 
    min_rating=0, 
    max_rating=10
):
    try:
        # genera la matrice user-item e i mappings
        mat, user_mapping, item_mapping, user_codes, item_codes = build_user_item_matrix(ratings)

        # converte user_id in stringa per consistenza
        user_id = str(user_id)
        # controlla se l'utente è nel dataset
        if user_id not in user_codes:
            # utente non presente nel dataset di training
            return []
        # indice numerico dell'utente nella matrice
        user_index = int(user_codes[user_id])

        # calcola il numero di item valutati per ogni utente
        counts = (mat != 0).sum(axis=1).A1
        # calcola la somma totale dei rating per ogni utente
        sums = np.array(mat.sum(axis=1)).flatten()
        # calcola la media dei rating per ogni utente, se non ha valutato nulla la media è 0
        user_means = np.where(counts > 0, sums / counts, 0.0)

        # genera una copia della matrice
        mat_centered = mat.copy().astype(float)
        # sottrae la media di ogni utente dai suoi rating (solo valori non zero)
        for u in range(mat_centered.shape[0]):
            # mat_centered.indptr[u] e mat_centered.indptr[u+1] danno gli indici di inizio e fine della riga u
            start, end = mat_centered.indptr[u], mat_centered.indptr[u+1]
            if start < end:
                # contiene solo i valori non zero della riga u
                mat_centered.data[start:end] -= float(user_means[u])

        # prende la riga dell'utente target (vettore dei suoi ratings centrati)
        target_row = mat_centered.getrow(user_index)
        # se l'utente non ha valutato nulla, non si possono fare raccomandazioni
        if target_row.nnz == 0:
            return []
        # calcola la similarità coseno tra l'utente target e tutti gli altri utenti
        # ritorna un array 1D con similarità
        similarities = cosine_similarity(target_row, mat_centered).flatten()

        # ordina gli utenti per similarità decrescente
        candidates = np.argsort(similarities)[::-1]
        neighbors = []
        for v in candidates:
            # ignora se è l'utente stesso
            if int(v) == int(user_index):
                continue
            # controlla il numero di item in comune valutati
            common = np.intersect1d(mat.getrow(user_index).nonzero()[1], mat.getrow(int(v)).nonzero()[1])
            # se non ha abbastanza item in comune (min_common), salta
            if len(common) < int(min_common):
                continue
            # altrimenti aggiunge il vicino alla lista
            neighbors.append(int(v))
            # se ha trovato abbastanza vicini (k), esce
            if len(neighbors) >= k:
                break
        # se non ha trovato vicini, non si possono fare raccomandazioni
        if not neighbors:
            return []

        # ottiene gli item già valutati dall'utente
        user_rated = set(mat.getrow(user_index).nonzero()[1].tolist())
        # tutti gli item nel dataframe
        all_items = range(mat.shape[1])
        # item da prevedere sono quelli non ancora valutati dall'utente
        items_to_predict = [int(i) for i in all_items if i not in user_rated]

        # calcola le predizioni
        preds = {}
        # scansiona gli item da prevedere
        for item in items_to_predict:
            num, den = 0.0, 0.0
            # accumula i contributi dei vicini
            for u in neighbors:
                # ottiene la riga del vicino come array
                row = mat.getrow(int(u)).toarray()[0]
                rating = float(row[int(item)])
                # considera solo se il vicino ha valutato l'item
                if rating != 0.0:
                    # somma pesata delle differenze dalla media (num e den)
                    sim = float(similarities[int(u)])
                    num += sim * (rating - float(user_means[int(u)]))
                    den += abs(sim)
            # se den > 0, ovvero se abbiamo almeno un vicino che ha valutato l'item
            if den > 0:
                # calcola la predizione come media pesata + media dell'utente
                pred = float(user_means[user_index]) + (num / den)
                # limita la predizione al range [min_rating, max_rating]
                pred = max(min_rating, min(max_rating, pred))
                preds[item] = pred
        # se non ci sono predizioni, ritorna lista vuota
        if not preds:
            return []

        # ordina le predizioni, e prende le top_n
        top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # converte gli indici degli item in ISBN originali
        recommendations = [(item_mapping[int(i)], float(score)) for i, score in top_items]
        return recommendations

    except Exception as e:
        # log error if using logger, otherwise print
        try:
            logger.error(f"User-based CF internal error for user {user_id}: {e}")
        except Exception:
            print(f"User-based CF internal error for user {user_id}: {e}")
        return []
