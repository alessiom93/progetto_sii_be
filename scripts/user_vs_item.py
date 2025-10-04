import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed  # ✅ cambiamo da Thread a Process

logger = logging.getLogger(__name__)

# -----------------------------
# Utility: build user-item matrix
# -----------------------------
def build_user_item_matrix(ratings: pd.DataFrame):
    ratings = ratings.copy()
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)

    user_cat = ratings['User-ID'].astype('category')
    item_cat = ratings['ISBN'].astype('category')

    user_mapping = dict(enumerate(user_cat.cat.categories))
    item_mapping = dict(enumerate(item_cat.cat.categories))
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))

    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_cat.cat.codes, item_cat.cat.codes)),
        shape=(len(user_mapping), len(item_mapping))
    )
    return mat, user_mapping, item_mapping, user_codes, item_codes

# -----------------------------
# User-based CF
# -----------------------------
def user_based_cf(user_id, ratings: pd.DataFrame, k=50, top_n=10, min_common=1, min_rating=0, max_rating=10):
    try:
        mat, user_mapping, item_mapping, user_codes, item_codes = build_user_item_matrix(ratings)
        user_id = str(user_id)
        if user_id not in user_codes:
            return []

        user_index = int(user_codes[user_id])
        counts = (mat != 0).sum(axis=1).A1
        sums = np.array(mat.sum(axis=1)).flatten()
        user_means = np.where(counts > 0, sums / counts, 0.0)

        mat_centered = mat.copy().astype(float)
        for u in range(mat_centered.shape[0]):
            start, end = mat_centered.indptr[u], mat_centered.indptr[u+1]
            if start < end:
                mat_centered.data[start:end] -= float(user_means[u])

        target_row = mat_centered.getrow(user_index)
        if target_row.nnz == 0:
            return []

        similarities = cosine_similarity(target_row, mat_centered).flatten()

        candidates = np.argsort(similarities)[::-1]
        neighbors = []
        for v in candidates:
            if int(v) == int(user_index):
                continue
            common = np.intersect1d(mat.getrow(user_index).nonzero()[1], mat.getrow(int(v)).nonzero()[1])
            if len(common) < int(min_common):
                continue
            neighbors.append(int(v))
            if len(neighbors) >= k:
                break

        if not neighbors:
            return []

        user_rated = set(mat.getrow(user_index).nonzero()[1].tolist())
        all_items = range(mat.shape[1])
        items_to_predict = [int(i) for i in all_items if i not in user_rated]

        preds = {}
        for item in items_to_predict:
            num, den = 0.0, 0.0
            for u in neighbors:
                row = mat.getrow(int(u)).toarray()[0]
                rating = float(row[int(item)])
                if rating != 0.0:
                    sim = float(similarities[int(u)])
                    num += sim * (rating - float(user_means[int(u)]))
                    den += abs(sim)
            if den > 0:
                pred = float(user_means[user_index]) + (num / den)
                pred = max(min_rating, min(max_rating, pred))
                preds[item] = pred

        if not preds:
            return []

        top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [(item_mapping[int(i)], float(score)) for i, score in top_items]
        return recommendations

    except Exception as e:
        logger.error(f"User-based CF internal error for user {user_id}: {e}")
        return []

# -----------------------------
# Item-based CF (semplice)
# -----------------------------
def item_based_cf(user_id, ratings: pd.DataFrame, k=50, top_n=10):
    try:
        mat, user_mapping, item_mapping, user_codes, item_codes = build_user_item_matrix(ratings)
        user_id = str(user_id)
        if user_id not in user_codes:
            return []

        user_index = int(user_codes[user_id])
        user_row = mat.getrow(user_index)
        rated_items = user_row.nonzero()[1]
        if len(rated_items) == 0:
            return []

        sims = cosine_similarity(mat.T, mat.T)
        scores = {}
        for item in rated_items:
            sim_items = np.argsort(sims[item])[::-1]
            count = 0
            for sim_item in sim_items:
                if sim_item == item:
                    continue
                if sim_item in rated_items:
                    continue
                scores[sim_item] = scores.get(sim_item, 0) + sims[item, sim_item]
                count += 1
                if count >= k:
                    break

        if not scores:
            return []

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [(item_mapping[int(i)], float(score)) for i, score in top_items]
        return recommendations

    except Exception as e:
        logger.error(f"Item-based CF internal error for user {user_id}: {e}")
        return []

# -----------------------------
# Campionamento denso/sparso
# -----------------------------
def sample_dense_subset(ratings_df, n_users=20, n_items=20, n_ratings=200, random_state=42):
    np.random.seed(random_state)

    top_users = ratings_df["User-ID"].value_counts().head(n_users).index
    top_items = ratings_df["ISBN"].value_counts().head(n_items).index

    subset = ratings_df[
        ratings_df["User-ID"].isin(top_users) & ratings_df["ISBN"].isin(top_items)
    ]

    max_possible = n_users * n_items
    n_ratings = min(n_ratings, max_possible, len(subset))

    subset = subset.sample(n=n_ratings, random_state=random_state)
    density = n_ratings / max_possible if max_possible > 0 else 0
    logger.debug(f"[DEBUG] Subset campionato: {subset.shape[0]} righe, {subset['User-ID'].nunique()} utenti, {subset['ISBN'].nunique()} item, densità={density:.2f}")

    return subset.reset_index(drop=True)

# -----------------------------
# Valutazione
# -----------------------------
def evaluate_recommendations(preds_dict, test_user_items, top_n=10):
    hit_count = 0
    precision_list, recall_list = [], []

    for u, recs in preds_dict.items():
        recs_top = [isbn for isbn, _ in recs[:top_n]]
        true_items = test_user_items.get(u, set())
        hits = len(set(recs_top) & true_items)
        hit_count += hits
        precision_list.append(hits / top_n if top_n > 0 else 0)
        recall_list.append(hits / len(true_items) if len(true_items) > 0 else 0)

    hit_rate = hit_count / sum(len(v) for v in test_user_items.values()) if test_user_items else 0
    precision = np.nanmean(precision_list) if precision_list else 0
    recall = np.nanmean(recall_list) if recall_list else 0

    return {"hit_rate": hit_rate, "precision": precision, "recall": recall}

# -----------------------------
# Helper: parallel execution con ProcessPoolExecutor
# -----------------------------
def parallel_recommendations(func, users, train_df, k, top_n, max_workers=16):
    """
    Parallelizza il calcolo delle raccomandazioni usando processi separati.
    Sicuro su Windows/macOS.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, u, train_df, k, top_n): u for u in users}
        for future in as_completed(futures):
            u = futures[future]
            try:
                results[str(u)] = future.result()
            except Exception as e:
                logger.warning(f"[ParallelProcess] Errore per utente {u}: {e}")
    return results

# -----------------------------
# Confronto algoritmi
# -----------------------------
def compare_user_item_cf(users_df, books_df, ratings_df, top_n=10, sample_n_users=500, k=200,
                         use_dense=True, n_users=500, n_items=500, n_ratings=150000,
                         use_parallel=True, max_workers=16):
    try:
        logger.info("Comparing user-based and item-based CF...")
        logger.info(f"Original ratings shape: {ratings_df.shape}")
        logger.info(f"Parameters: top_n={top_n}, sample_n_users={sample_n_users}, k={k}, use_dense={use_dense}, n_users={n_users}, n_items={n_items}, n_ratings={n_ratings}, use_parallel={use_parallel}, max_workers={max_workers}")

        if use_dense:
            ratings_df = sample_dense_subset(ratings_df, n_users=n_users, n_items=n_items, n_ratings=n_ratings)

        all_users = ratings_df['User-ID'].unique()
        sampled_users = np.random.choice(all_users, size=min(sample_n_users, len(all_users)), replace=False)

        test_user_items = {}
        for uid in sampled_users:
            user_ratings = ratings_df[ratings_df['User-ID'] == uid]
            if len(user_ratings) < 2:
                continue
            test_items = set(user_ratings.sample(frac=0.2, random_state=42)['ISBN'])
            train_ratings = ratings_df.drop(user_ratings[user_ratings['ISBN'].isin(test_items)].index)
            test_user_items[str(uid)] = test_items

        if use_parallel:
            preds_user = parallel_recommendations(user_based_cf, sampled_users, ratings_df, k, top_n, max_workers)
            preds_item = parallel_recommendations(item_based_cf, sampled_users, ratings_df, k, top_n, max_workers)
        else:
            preds_user = {str(u): user_based_cf(u, ratings_df, k=k, top_n=top_n) for u in sampled_users}
            preds_item = {str(u): item_based_cf(u, ratings_df, k=k, top_n=top_n) for u in sampled_users}

        user_metrics = evaluate_recommendations(preds_user, test_user_items, top_n=top_n)
        item_metrics = evaluate_recommendations(preds_item, test_user_items, top_n=top_n)

        results = {"user_based": user_metrics, "item_based": item_metrics}
        logger.debug(f"Comparison results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error comparing user-based and item-based CF: {e}")
        return {}
