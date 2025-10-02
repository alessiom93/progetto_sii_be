import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_user_item_matrix(ratings: pd.DataFrame):
    ratings = ratings.copy()
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    user_cat = ratings['User-ID'].astype('category')
    item_cat = ratings['ISBN'].astype('category')
    user_mapping = dict(enumerate(user_cat.cat.categories))   # index -> User-ID
    item_mapping = dict(enumerate(item_cat.cat.categories))   # index -> ISBN
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))  # User-ID -> index
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))  # ISBN -> index

    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_cat.cat.codes, item_cat.cat.codes)),
        shape=(len(user_mapping), len(item_mapping))
    )
    return mat, user_mapping, item_mapping, user_codes, item_codes

def user_based_cf(user_id, ratings: pd.DataFrame, k=50, top_n=10, min_common=1, min_rating=0, max_rating=10):
    """
    Versione robusta user-based CF.
    Ritorna list[(ISBN, predicted_rating)] oppure [] se non possibile.
    """
    try:
        # build matrix + mappings
        mat, user_mapping, item_mapping, user_codes, item_codes = build_user_item_matrix(ratings)

        # ensure user_id is string and exists in codes
        user_id = str(user_id)
        if user_id not in user_codes:
            # utente non presente nel dataset di training
            return []

        user_index = int(user_codes[user_id])  # indice numerico

        # safe user means: avoid div by zero
        counts = (mat != 0).sum(axis=1).A1
        sums = np.array(mat.sum(axis=1)).flatten()
        user_means = np.where(counts > 0, sums / counts, 0.0)

        # center only non-empty rows
        mat_centered = mat.copy().astype(float)
        for u in range(mat_centered.shape[0]):
            start, end = mat_centered.indptr[u], mat_centered.indptr[u+1]
            if start < end:
                mat_centered.data[start:end] -= float(user_means[u])

        # cosine similarity (centered => approx Pearson)
        target_row = mat_centered.getrow(user_index)
        if target_row.nnz == 0:
            return []   # user senza rating nel train
        similarities = cosine_similarity(target_row, mat_centered).flatten()

        # candidate neighbors sorted by similarity desc, skip self
        candidates = np.argsort(similarities)[::-1]
        neighbors = []
        for v in candidates:
            if int(v) == int(user_index):
                continue
            # check min common items if requested
            common = np.intersect1d(mat.getrow(user_index).nonzero()[1], mat.getrow(int(v)).nonzero()[1])
            if len(common) < int(min_common):
                continue
            neighbors.append(int(v))
            if len(neighbors) >= k:
                break

        if not neighbors:
            return []

        # items to predict: indices where user has 0
        user_rated = set(mat.getrow(user_index).nonzero()[1].tolist())
        all_items = range(mat.shape[1])
        items_to_predict = [int(i) for i in all_items if i not in user_rated]

        preds = {}
        for item in items_to_predict:
            num, den = 0.0, 0.0
            # accumulate using dense row from neighbors
            for u in neighbors:
                # safe access: take neighbor row as dense
                row = mat.getrow(int(u)).toarray()[0]
                rating = float(row[int(item)])
                if rating != 0.0:
                    sim = float(similarities[int(u)])
                    num += sim * (rating - float(user_means[int(u)]))
                    den += abs(sim)
            if den > 0:
                pred = float(user_means[user_index]) + (num / den)
                # clip to rating range
                pred = max(min_rating, min(max_rating, pred))
                preds[item] = pred

        if not preds:
            return []

        # sort top_n and map index -> ISBN
        top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [(item_mapping[int(i)], float(score)) for i, score in top_items]
        return recommendations

    except Exception as e:
        # log error if using logger, otherwise print
        try:
            logger.error(f"User-based CF internal error for user {user_id}: {e}")
        except Exception:
            print(f"User-based CF internal error for user {user_id}: {e}")
        return []
