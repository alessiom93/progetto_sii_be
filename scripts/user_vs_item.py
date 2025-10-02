# compare_cf_models.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --------------------------------------------
# FUNZIONI COMUNI
# --------------------------------------------

def build_user_item_matrix(ratings: pd.DataFrame):
    """Costruisce la matrice utente x item (sparse) dai rating."""
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)

    user_ids = ratings['User-ID'].astype('category')
    item_ids = ratings['ISBN'].astype('category')

    user_mapping = dict(enumerate(user_ids.cat.categories))
    item_mapping = dict(enumerate(item_ids.cat.categories))

    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_ids.cat.codes, item_ids.cat.codes))
    )

    return mat, user_mapping, item_mapping

# --------------------------------------------
# USER-BASED CF
# --------------------------------------------

def user_based_cf(user_id, ratings, k=50, top_n=10):
    """
    Raccomandazioni User-based CF con gestione errori.

    Se l'utente non ha vicini validi o non si possono predire libri,
    ritorna lista vuota invece di crashare.
    """
    try:
        # Costruisco matrice utente × item
        mat, user_mapping, item_mapping = build_user_item_matrix(ratings)

        # Mappa inversa
        index_to_user = {v: k for k, v in user_mapping.items()}
        index_to_item = {v: k for k, v in item_mapping.items()}

        user_id = str(user_id)
        if user_id not in index_to_user:
            return []  # utente non trovato

        user_index = index_to_user[user_id]

        # Media rating di ogni utente
        user_means = np.array(mat.sum(axis=1)).flatten() / np.maximum((mat != 0).sum(axis=1).A1, 1)

        # Centriamo i rating
        mat_centered = mat.copy().astype(float)
        for u in range(mat.shape[0]):
            start, end = mat_centered.indptr[u], mat_centered.indptr[u + 1]
            if start < end:
                mat_centered.data[start:end] -= user_means[u]

        # Similarità coseno
        similarities = cosine_similarity(mat_centered[user_index], mat_centered).flatten()

        # Vicini più simili
        similar_users = np.argsort(similarities)[::-1]
        similar_users = [u for u in similar_users if u != user_index][:k]

        # Libri non valutati
        user_rated_items = mat[user_index].nonzero()[1]
        all_items = set(range(mat.shape[1]))
        items_to_predict = list(all_items - set(user_rated_items))

        preds = {}
        for item in items_to_predict:
            num, den = 0.0, 0.0
            for u in similar_users:
                rating = mat[u, item]
                if rating != 0:
                    sim = float(similarities[u])
                    num += sim * (rating - user_means[u])
                    den += abs(sim)
            if den > 0:
                preds[item] = user_means[user_index] + num / den

        if not preds:
            return []

        top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [(index_to_item[i], score) for i, score in top_items]

        return recommendations

    except Exception as e:
        # Qui logghi, ma non blocchi il processo
        print(f"User-based CF error for user {user_id}: {e}")
        return []


# --------------------------------------------
# ITEM-BASED CF
# --------------------------------------------

def item_based_cf(user_id, ratings, k=50, top_n=10):
    """
    Raccomandazioni Item-based CF con gestione errori.

    Se non si trovano item simili o ci sono inconsistenze,
    ritorna lista vuota invece di crashare.
    """
    try:
        mat, user_mapping, item_mapping = build_user_item_matrix(ratings)

        index_to_user = {v: k for k, v in user_mapping.items()}
        index_to_item = {v: k for k, v in item_mapping.items()}

        user_id = str(user_id)
        if user_id not in index_to_user:
            return []

        user_index = index_to_user[user_id]

        # Indici degli item valutati dall'utente
        user_rated_items = mat[user_index].nonzero()[1]
        if len(user_rated_items) == 0:
            return []

        # Similarità item × item
        similarities = cosine_similarity(mat.T, mat.T)

        preds = {}
        for item in range(mat.shape[1]):
            if item in user_rated_items:
                continue

            num, den = 0.0, 0.0
            for rated_item in user_rated_items:
                if rated_item >= similarities.shape[0]:
                    continue  # skip se indice fuori range

                sim = similarities[item, rated_item]
                rating = mat[user_index, rated_item]

                if sim > 0 and rating > 0:
                    num += sim * rating
                    den += sim

            if den > 0:
                preds[item] = num / den

        if not preds:
            return []

        top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [(index_to_item[i], score) for i, score in top_items]

        return recommendations

    except Exception as e:
        print(f"Item-based CF error for user {user_id}: {e}")
        return []


# --------------------------------------------
# FUNZIONE DI VALUTAZIONE
# --------------------------------------------

import numpy as np
from sklearn.metrics import precision_score, recall_score

def compare_user_item_cf(ratings, top_n=10, sample_size=1000, seed=42):
    """
    Confronta User-based e Item-based CF usando dati di test.

    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame con ['User-ID', 'ISBN', 'Book-Rating']
    top_n : int
        Numero di raccomandazioni da generare
    sample_size : int
        Numero massimo di utenti da testare (per velocizzare)
    seed : int
        Random seed per riproducibilità

    Returns
    -------
    dict
        {
            "user_based": {"precision": float, "recall": float},
            "item_based": {"precision": float, "recall": float}
        }
    """

    np.random.seed(seed)
    unique_users = ratings['User-ID'].unique()
    sampled_users = np.random.choice(unique_users, min(sample_size, len(unique_users)), replace=False)

    user_precisions = []
    user_recalls = []
    item_precisions = []
    item_recalls = []

    # Iteriamo sugli utenti campionati
    for user_id in sampled_users:
        # --- Dati di test: prendiamo gli ultimi 10 rating dell'utente ---
        user_ratings = ratings[ratings['User-ID'] == user_id].sort_values('Book-Rating')
        test_books = user_ratings['ISBN'].values[-top_n:]
        if len(test_books) < 2:
            continue  # salta utenti troppo pochi rating

        test_set = set(test_books)

        # --- Raccomandazioni ---
        user_recs = user_based_cf(user_id, ratings, top_n=top_n)
        item_recs = item_based_cf(user_id, ratings, top_n=top_n)

        # --- Pred/ground truth binarizzati ---
        def binarize(recs):
            rec_books = set([isbn for isbn, _ in recs])
            y_true = [1 if b in test_set else 0 for b in rec_books]
            y_pred = [1] * len(y_true)  # tutte le raccomandazioni sono considerate positive
            return y_true, y_pred

        if user_recs:
            y_true, y_pred = binarize(user_recs)
            user_precisions.append(precision_score(y_true, y_pred))
            user_recalls.append(recall_score(y_true, y_pred))

        if item_recs:
            y_true, y_pred = binarize(item_recs)
            item_precisions.append(precision_score(y_true, y_pred))
            item_recalls.append(recall_score(y_true, y_pred))

    return {
        "user_based": {
            "precision": np.mean(user_precisions) if user_precisions else 0.0,
            "recall": np.mean(user_recalls) if user_recalls else 0.0
        },
        "item_based": {
            "precision": np.mean(item_precisions) if item_precisions else 0.0,
            "recall": np.mean(item_recalls) if item_recalls else 0.0
        }
    }

