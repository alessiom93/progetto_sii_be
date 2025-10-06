import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from item_based_cf_rs import item_based_cf
from user_based_cf_rs import user_based_cf

logger = logging.getLogger(__name__)

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
def compare_user_item_cf(ratings_df, top_n=10, sample_n_users=500, k=200,
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
