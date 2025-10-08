import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

''' Item-Based Collaborative Filtering Recommender System 
1. Costruzione della matrice: crea una matrice items x users.
2. Normalizzazione: calcola la media dei rating per utente e centra i dati.
3. Calcolo similarità: trova libri simili usando la cosine similarity.
4. Selezione vicini: mantiene solo i k items più simili all'item target che hanno abbastanza utenti in comune.
5. Predizione: usa una media pesata delle differenze dai rating medi dell'utente target.
6. Output: restituisce i top_n libri con rating predetto più alto.
'''

# costruisce la matrice item-user
# input: dataframe con colonne User-ID, ISBN, Book-Rating
# output: matrice item-user + mappings tra indici e ID originali
def build_item_user_matrix(ratings: pd.DataFrame):
    # copia del dataframe su cui lavorare
    ratings = ratings.copy()
    # conversione degli ID in stringhe per consistenza
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    # conversione dei valori in categorie, ovvero in indici numerici, perchè più efficienti
    item_cat = ratings['ISBN'].astype('category')
    user_cat = ratings['User-ID'].astype('category')
    # mapping tra indici numerici e ID originali
    item_mapping = dict(enumerate(item_cat.cat.categories))
    user_mapping = dict(enumerate(user_cat.cat.categories))
    # mapping inversi da ID a indici numerici
    item_codes = dict(zip(item_cat.cat.categories, item_cat.cat.codes))
    user_codes = dict(zip(user_cat.cat.categories, user_cat.cat.codes))
    # costruzione matrice item-user, righe=items, colonne=users, valori=ratings
    # La matrice CSR è rappresentata da tre array: data (i ratings non nulli), indices (indici colonna della posizione dei ratings), indptr (indici di inizio/fine di ogni riga)
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float), # valori dei rating
         (item_cat.cat.codes, user_cat.cat.codes)), # coordinate (riga, colonna)
        shape=(len(item_mapping), len(user_mapping)) # dimensioni (n_items, n_users)
    )
    # ritorna matrice + mappings
    return mat, item_mapping, user_mapping, item_codes, user_codes

# item-based collaborative filtering
# input: user_id, dataframe con colonne User-ID, ISBN, Book-Rating
#        k=numero di items vicini, top_n=numero di raccomandazioni, min_common=minimo numero di utenti in comune tra item
#        min_rating, max_rating = range dei ratings
# output: lista di tuple (ISBN, predicted_rating) ordinate per predicted_rating decrescente
#         oppure lista vuota se non possibile fare raccomandazioni
def item_based_cf(
    user_id, 
    ratings: pd.DataFrame, 
    k=25, 
    top_n=10, 
    min_common=15, 
    min_rating=0, 
    max_rating=10
):
    try:
      # genera la matrice item-user e i mappings
      mat, item_mapping, user_mapping, item_codes, user_codes = build_item_user_matrix(ratings)

      # converte user_id in stringa per consistenza
      user_id = str(user_id)
      # controlla se l'utente è nel dataset
      if user_id not in user_codes:
          # utente non presente nel dataset di training
          return []
      # indice numerico dell'utente nella matrice
      user_index = int(user_codes[user_id])

      # calcola il numero di item valutati per ogni utente
      counts = (mat != 0).sum(axis=0).A1
      # calcola la somma totale dei rating per ogni utente
      sums = np.array(mat.sum(axis=0)).flatten()
      # calcola la media dei rating per ogni utente, se non ha valutato nulla la media è 0
      user_means = np.where(counts > 0, sums / counts, 0.0)

      # crea una copia della matrice per centrare i rating
      # centro per utente (colonna) → adjusted cosine similarity
      mat_csc = mat.tocsc().astype(float)
      # sottrae la media di ogni utente dai suoi rating (solo valori non zero)
      for u in range(mat_csc.shape[1]):
          # mat_csc.indptr[u] e mat_csc.indptr[u+1] danno gli indici di inizio e fine della riga u
          start, end = mat_csc.indptr[u], mat_csc.indptr[u+1]
          if start < end:
              # contiene solo i valori non zero della riga u
              mat_csc.data[start:end] -= float(user_means[u])
      # torno a CSR (righe=item) per calcolare la similarità sulle righe
      mat_centered = mat_csc.tocsr()

      # calcolo similarità tra tutti gli item (libri)
      # cosine similarity sulla matrice item × user
      # ogni cella (i, j) = similarità coseno tra libro i e libro j
      item_similarities = cosine_similarity(mat_centered)

      # trovo tutti i libri già valutati dall’utente
      user_rated_items = mat[:, user_index].nonzero()[0]

      # calcola le predizioni
      preds = {}
      # scansiona gli item su cui predire (non valutati dall'utente)
      for item in range(mat.shape[0]):
          # skip libri già valutati
          if item in user_rated_items:
              continue
          # conterrà gli item già valutati dall'utente che hanno abbastanza utenti in comune con l'item target (i k più)
          valid_items = []
          valid_sims = []
          # scansiona i libri già valutati dall'utente
          for rated_item in user_rated_items:
              # trova gli utenti che hanno valutato l'item e il rated_item
              # item = indice del libro da predire
              # rated_item = indice del libro già valutato dall'utente
              users_item = set(mat[item, :].nonzero()[1])
              users_rated_item = set(mat[rated_item, :].nonzero()[1])
              common_users = users_item & users_rated_item
              # considera solo item che hanno almeno min_common utenti in comune
              if len(common_users) >= min_common:
                  sim = item_similarities[item, rated_item]
                  # considera questo item come valido per la predizione (simile e con utenti in comune)
                  valid_items.append(rated_item)
                  valid_sims.append(sim)
          # se non ci sono items validi, salta
          if not valid_items:
              continue

          # prende i k items più simili
          top_k_idx = np.argsort(valid_sims)[::-1][:k]
          # indici dei libri simili
          top_similar_items = [valid_items[i] for i in top_k_idx]
          # valori di similarità corrispondenti
          top_similarities = [valid_sims[i] for i in top_k_idx]

          # Rating predetto come media pesata
          num, den = 0.0, 0.0
          # scansiona i libri simili
          for sim_item, sim in zip(top_similar_items, top_similarities):
              # prendo il rating dato dall'utente a quel libro
              rating = mat[sim_item, user_index].item()
              # considera solo rating non nulli
              if rating != 0:
                  # somma pesata delle deviazioni dalla media
                  num += sim * (rating - float(user_means[user_index]))
                  den += abs(sim)
          # se abbiamo almeno una similarità non nulla, calcola la predizione
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
      recommendations = [(item_mapping[i], score) for i, score in top_items]
      return recommendations
    
    except Exception as e:
        # log error if using logger, otherwise print
        try:
            logger.error(f"User-based CF internal error for user {user_id}: {e}")
        except Exception:
            print(f"User-based CF internal error for user {user_id}: {e}")
        return []
