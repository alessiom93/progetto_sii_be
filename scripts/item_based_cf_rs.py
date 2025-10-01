# Creo una matrice item × user: righe = libri, colonne = utenti, valori = rating.
# Centratura dei rating: sottraggo la media di ciascun libro → Pearson similarity ≈ cosine similarity dei dati centrati.
# Similarità item-item: calcolo la coseno tra libri letti dall’utente e tutti gli altri libri.
# Top-k vicini: consideriamo solo i k libri più simili.
# Predizione rating: media libro + media pesata dei rating dei libri simili.
# Output: top-N libri raccomandati non ancora letti dall’utente target.

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_item_user_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice item × user (sparse matrix) a partire dal dataframe ratings.
    Ogni riga = un libro, ogni colonna = un utente, ogni cella = rating dato dall’utente a quel libro.

    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame con almeno 3 colonne: 'User-ID', 'ISBN', 'Book-Rating'.

    Returns
    -------
    mat : csr_matrix
        Matrice sparsa item × user con i rating.
    item_mapping : dict
        Dizionario che mappa indici interi → ISBN originali.
    user_mapping : dict
        Dizionario che mappa indici interi → ID utente originali.
    """

    # Convertiamo 'User-ID' e 'ISBN' in categorie numeriche
    user_ids = ratings['User-ID'].astype('category')
    item_ids = ratings['ISBN'].astype('category')

    # Dizionari per tornare da indici numerici agli ID originali
    user_mapping = dict(enumerate(user_ids.cat.categories))
    item_mapping = dict(enumerate(item_ids.cat.categories))

    # Creiamo matrice sparsa item × user
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (item_ids.cat.codes, user_ids.cat.codes))
    )

    return mat, item_mapping, user_mapping


def item_based_cf(user_id, ratings, k=50, top_n=10):
    """
    Sistema di raccomandazione Item-based CF con Pearson correlation.
    L’idea è:
        - trovare libri simili a quelli già valutati dall’utente
        - predire rating sui libri non ancora letti
        - restituire i top-N libri raccomandati

    Parameters
    ----------
    user_id : int o str
        ID dell’utente target.
    ratings : pd.DataFrame
        DataFrame con colonne ['User-ID', 'ISBN', 'Book-Rating'].
    k : int, default=50
        Numero di item simili da considerare per ciascun libro.
    top_n : int, default=10
        Numero di libri da raccomandare.

    Returns
    -------
    recommendations : list of tuples
        Lista di tuple (ISBN, punteggio previsto), ordinate per punteggio decrescente.
    """

    # 1. Costruiamo la matrice item × user
    mat, item_mapping, user_mapping = build_item_user_matrix(ratings)

    # 2. Troviamo l’indice numerico dell’utente target
    user_index = list(user_mapping.keys())[list(user_mapping.values()).index(user_id)]

    # 3. Individuiamo tutti i libri valutati dall’utente target
    rated_items = mat[:, user_index].nonzero()[0]  # righe = libri

    # 4. Centriamo i rating di ciascun libro sulla media del libro (Pearson)
    #    In Item-based CF la Pearson similarity è cosine dei rating centrati
    item_means = np.array(mat.sum(axis=1)).flatten() / (mat != 0).sum(axis=1).A1
    mat_centered = mat.copy().astype(float)
    for i in range(mat.shape[0]):
        cols = mat_centered[i].nonzero()[1]  # colonne = utenti che hanno valutato il libro
        mat_centered[i, cols] -= item_means[i]

    # 5. Calcoliamo la similarità coseno tra tutti i libri valutati dall’utente e tutti gli altri libri
    #    Useremo queste similarità per predire nuovi rating
    #    La matrice di similarità può essere calcolata solo per le righe interessate
    recommendations_dict = {}
    for item_idx in rated_items:
        similarities = cosine_similarity(mat_centered[item_idx], mat_centered).flatten()

        # 6. Ordiniamo i libri per similarità decrescente
        similar_items_idx = np.argsort(similarities)[::-1]

        # 7. Prendiamo solo i primi k libri più simili (escluso il libro stesso)
        similar_items_idx = [i for i in similar_items_idx if i != item_idx][:k]

        # 8. Predizione rating per ciascun libro non ancora valutato dall’utente
        for sim_item_idx in similar_items_idx:
            if sim_item_idx in rated_items:
                continue  # Skip libri già letti
            sim = similarities[sim_item_idx]
            # Weight = similarità * (rating libro simile - media libro simile)
            pred = sim * (mat[sim_item_idx, user_index] - item_means[sim_item_idx])
            if sim_item_idx in recommendations_dict:
                recommendations_dict[sim_item_idx]['numerator'] += pred
                recommendations_dict[sim_item_idx]['denominator'] += abs(sim)
            else:
                recommendations_dict[sim_item_idx] = {'numerator': pred, 'denominator': abs(sim)}

    # 9. Calcoliamo il rating previsto finale per ciascun libro
    predictions = []
    for item_idx, vals in recommendations_dict.items():
        if vals['denominator'] > 0:
            predicted_rating = item_means[item_idx] + vals['numerator'] / vals['denominator']
            predictions.append((item_idx, predicted_rating))

    # 10. Ordiniamo le predizioni e prendiamo le top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    # 11. Convertiamo gli indici in ISBN originali
    inv_item_mapping = {v: k for k, v in item_mapping.items()}
    recommendations = [(inv_item_mapping[i], score) for i, score in top_predictions]

    return recommendations

# Esempio di utilizzo:
ratings = pd.read_csv("ratings_explicit_mod.csv")

recommendations = item_based_cf(user_id=276729, ratings=ratings, k=50, top_n=10)

print("Raccomandazioni Item-based per l’utente 276729:")
for isbn, score in recommendations:
    print(isbn, score)
