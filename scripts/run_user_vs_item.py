import pandas as pd
from user_vs_item import compare_user_item_cf
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    start_time = time.time()

    # Carica i CSV
    ratings_df = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
    users_df = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/users_mod.csv')
    books_df = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/books_mod.csv')

    # Parametri personalizzati
    results = compare_user_item_cf(
        users_df, books_df, ratings_df,
        top_n=20,
        sample_n_users=100,
        k=200,
        use_dense=True,
        n_users=50,
        n_items=50,
        n_ratings=2000,
        use_parallel=True,
        max_workers=16  # sfrutta tutti i core disponibili
    )

    end_time = time.time()
    print(f"\n--- Risultati confronto User vs Item CF ---")
    print(results)
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")
