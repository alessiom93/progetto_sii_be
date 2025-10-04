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
        top_n=50, # numero di raccomandazioni per utente
        sample_n_users=400, # numero di utenti campionati per il test
        k=400, # numero di vicini considerati
        use_dense=True,
        n_users=700, # numero di utenti nel sotto-dataset denso
        n_items=700, # numero di libri nel sotto-dataset denso
        n_ratings=int(0.85 * 700 * 700), # numero di valutazioni nel sotto-dataset denso
        use_parallel=True,
        max_workers=16  # sfrutta tutti i core disponibili
    )

    end_time = time.time()
    print(f"\n--- Risultati confronto User vs Item CF ---")
    print(results)
    print(f"\nTempo di esecuzione: {end_time - start_time:.2f} secondi")
