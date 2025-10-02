import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_item_user_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice item × user (libri × utenti) come csr_matrix.
    
    Parametri
    ----------
    ratings : pd.DataFrame
        Deve contenere almeno ['User-ID', 'ISBN', 'Book-Rating']
    
    Ritorna
    -------
    mat : csr_matrix
        Matrice sparsa item × user
    item_mapping : dict
        Mappa indice numerico -> ISBN
    user_mapping : dict
        Mappa indice numerico -> User-ID
    item_codes : dict
        Mappa ISBN -> indice numerico
    user_codes : dict
        Mappa User-ID -> indice numerico
    """
    ratings = ratings.copy()
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)

    item_cat = ratings['ISBN'].astype('category')
    user_cat = ratings['User-ID'].astype('category')

    item_mapping = dict(enumerate(item_cat.cat.categories))
    user_mapping = dict(enumerate(user_cat.cat.categories))
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))

    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (item_cat.cat.codes, user_cat.cat.codes)),
        shape=(len(item_mapping), len(user_mapping))
    )

    return mat, item_mapping, user_mapping, item_codes, user_codes


def item_based_cf(
    user_id, 
    ratings: pd.DataFrame, 
    k=50, 
    top_n=10, 
    min_common=1, 
    min_rating=0, 
    max_rating=10
):
    """
    Sistema di raccomandazione Item-based CF con Pearson correlation.
    Parallelo a user_based_cf.

    Parametri
    ----------
    user_id : int o str
        ID dell’utente target
    ratings : pd.DataFrame
        DataFrame con colonne ['User-ID', 'ISBN', 'Book-Rating']
    k : int
        Numero di libri simili da considerare
    top_n : int
        Numero di libri da raccomandare
    min_common : int
        Minimo numero di utenti in comune tra due libri per calcolare similarità
    min_rating : float
        Valore minimo del rating
    max_rating : float
        Valore massimo del rating

    Ritorna
    -------
    recommendations : list of tuples
        Lista di (ISBN, rating previsto) ordinata per rating decrescente
    """
    # 1️⃣ Costruisco matrice item × user
    mat, item_mapping, user_mapping, item_codes, user_codes = build_item_user_matrix(ratings)

    # 2️⃣ Controllo che l'utente esista
    user_id = str(user_id)
    if user_id not in user_codes:
        raise ValueError(f"User ID {user_id} non presente nel dataset")
    user_index = user_codes[user_id]

    # 3️⃣ Trovo tutti i libri già valutati dall’utente
    user_rated_items = mat[:, user_index].nonzero()[0]

    # 4️⃣ Calcolo similarità tra tutti gli item (libri)
    #    Cosine similarity sulla matrice item × user
    item_similarities = cosine_similarity(mat)

    # 5️⃣ Predizione dei rating
    preds = {}
    for item in range(mat.shape[0]):
        if item in user_rated_items:
            continue  # skip libri già valutati
        # Trovo i libri più simili che l'utente ha già valutato
        sims = item_similarities[item, user_rated_items]
        if len(sims) < min_common:
            continue  # skip se non ci sono abbastanza utenti in comune
        top_k_idx = np.argsort(sims)[::-1][:k]
        top_similar_items = user_rated_items[top_k_idx]
        top_similarities = sims[top_k_idx]

        # Rating predetto come media pesata
        num, den = 0.0, 0.0
        for sim_item, sim in zip(top_similar_items, top_similarities):
            rating = mat[sim_item, user_index]
            if rating != 0:
                num += sim * rating
                den += abs(sim)
        if den > 0:
            pred_rating = num / den
            # Clipping ai valori min/max
            pred_rating = max(min_rating, min(max_rating, pred_rating))
            preds[item] = pred_rating

    # 6️⃣ Ordino e prendo top N
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # 7️⃣ Mappo indici numerici in ISBN originali
    recommendations = [(item_mapping[i], score) for i, score in top_items]

    return recommendations
