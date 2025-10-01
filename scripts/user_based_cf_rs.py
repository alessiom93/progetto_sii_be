import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice utente x item (sparse) e mapping sicuri tra User-ID / ISBN e indici numerici.
    """
    # Convertiamo ID a stringa e creiamo categorie
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)

    user_cat = ratings['User-ID'].astype('category')
    item_cat = ratings['ISBN'].astype('category')

    # Mappatura sicura
    user_mapping = dict(enumerate(user_cat.cat.categories))  # indice -> User-ID
    item_mapping = dict(enumerate(item_cat.cat.categories))  # indice -> ISBN
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))  # User-ID -> indice
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))  # ISBN -> indice

    # Matrice sparsa
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_cat.cat.codes, item_cat.cat.codes))
    )

    return mat, user_mapping, item_mapping, user_codes, item_codes

def user_based_cf(user_id, ratings, k=50, top_n=10):
    """
    User-based CF sicuro con cosine similarity su matrice centrata.
    """
    mat, user_mapping, item_mapping, user_codes, item_codes = build_user_item_matrix(ratings)

    user_id = str(user_id)
    if user_id not in user_codes:
        raise ValueError(f"User ID {user_id} non presente nel dataset")

    user_index = user_codes[user_id]

    # Media dei rating per utente
    user_means = np.array(mat.sum(axis=1)).flatten() / (mat != 0).sum(axis=1).A1

    # Centriamo la matrice rispetto alla media utente
    mat_centered = mat.copy().astype(float)
    for u in range(mat.shape[0]):
        start = mat_centered.indptr[u]
        end = mat_centered.indptr[u+1]
        if start < end:
            mat_centered.data[start:end] -= user_means[u]

    # Similarità coseno tra utente target e tutti gli altri
    similarities = cosine_similarity(mat_centered[user_index], mat_centered).flatten()

    # Ordiniamo e prendiamo i k vicini più simili (escludendo se stesso)
    similar_users = np.argsort(similarities)[::-1]
    similar_users = [u for u in similar_users if u != user_index][:k]

    # Libri non ancora valutati
    user_rated_items = mat[user_index].nonzero()[1]
    all_items = set(range(mat.shape[1]))
    items_to_predict = list(all_items - set(user_rated_items))

    # Predizione dei rating
    preds = {}
    for item in items_to_predict:
        num, den = 0.0, 0.0
        for u in similar_users:
            rating = mat[u, item]
            if rating != 0:
                num += similarities[u] * (rating - user_means[u])
                den += abs(similarities[u])
        if den > 0:
            preds[item] = user_means[user_index] + num / den

    # Ordino e prendo top N
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Converto indici in ISBN originali
    recommendations = [(item_mapping[i], score) for i, score in top_items]

    return recommendations