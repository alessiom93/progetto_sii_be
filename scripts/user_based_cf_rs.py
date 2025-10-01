# Creo una matrice sparsa users × items con i rating (scipy.sparse).
# Normalizzo i rating (centrati sulla media dell’utente, come richiede la Pearson correlation).
# Calcolo le similarità tra utenti in modo vettoriale usando cosine similarity (che equivale a Pearson se i dati sono centrati sulla media).
# Predico i rating usando i K vicini più simili.
# Predizione: rating medio dell’utente + media pesata dei rating dei vicini.
# Output: lista ordinata di libri non ancora letti con rating previsto.

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice utente × item (sparse matrix) a partire dal dataframe ratings.
    Ogni riga = un utente, ogni colonna = un libro, ogni cella = rating dato dall'utente a quel libro.

    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame con almeno 3 colonne: 'User-ID', 'ISBN', 'Book-Rating'.

    Returns
    -------
    mat : csr_matrix
        Matrice sparsa (Compressed Sparse Row) utente × item con i rating.
    user_mapping : dict
        Dizionario che mappa indici interi → ID utente originali.
    item_mapping : dict
        Dizionario che mappa indici interi → ISBN originali.
    """

    # Convertiamo 'User-ID' e 'ISBN' in categorie numeriche (codici da 0 a N-1).
    user_ids = ratings['User-ID'].astype('category')
    item_ids = ratings['ISBN'].astype('category')

    # Creiamo i dizionari per tornare dagli indici numerici agli ID originali.
    user_mapping = dict(enumerate(user_ids.cat.categories))
    item_mapping = dict(enumerate(item_ids.cat.categories))

    # Creiamo la matrice sparsa utente × item
    # - Righe = utenti (user_ids.cat.codes)
    # - Colonne = libri (item_ids.cat.codes)
    # - Valori = rating (Book-Rating)
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_ids.cat.codes, item_ids.cat.codes))
    )

    return mat, user_mapping, item_mapping


def user_based_cf(user_id, ratings, k=50, top_n=10):
    """
    Sistema di raccomandazione User-based CF con Pearson correlation.

    Parameters
    ----------
    user_id : int o str
        ID dell’utente target (quello a cui vogliamo fare le raccomandazioni).
    ratings : pd.DataFrame
        DataFrame con colonne ['User-ID', 'ISBN', 'Book-Rating'].
    k : int, default=50
        Numero di utenti simili da considerare (vicini più simili).
    top_n : int, default=10
        Numero di libri da raccomandare.

    Returns
    -------
    recommendations : list of tuples
        Lista di tuple (ISBN, punteggio previsto), ordinate per punteggio decrescente.
    """

    # 1. Costruiamo la matrice utente × item
    mat, user_mapping, item_mapping = build_user_item_matrix(ratings)

    # 2. Troviamo l'indice numerico corrispondente all'utente target
    # (perché nella matrice usiamo indici da 0 a N-1, non gli ID originali).
    user_index = list(user_mapping.keys())[list(user_mapping.values()).index(user_id)]

    # 3. Calcoliamo la media dei rating per ogni utente
    # (serve per centrare i rating rispetto alla media, come richiede la Pearson).
    user_means = np.array(mat.sum(axis=1)).flatten() / (mat != 0).sum(axis=1).A1

    # 4. Creiamo una copia della matrice centrata rispetto alla media dell’utente
    # → questo trasforma la cosine similarity in Pearson correlation.
    mat_centered = mat.copy().astype(float)
    for u in range(mat.shape[0]):
        rows = mat_centered[u].nonzero()[1]  # colonne (libri) valutati dall’utente u
        mat_centered[u, rows] -= user_means[u]

    # 5. Calcoliamo la similarità coseno tra l’utente target e tutti gli altri
    # - Otteniamo un vettore di similarità (uno per ogni utente).
    similarities = cosine_similarity(mat_centered[user_index], mat_centered).flatten()

    # 6. Ordiniamo gli utenti per similarità decrescente
    similar_users = np.argsort(similarities)[::-1]

    # 7. Escludiamo l’utente stesso (perché sarebbe sempre identico a sé stesso)
    # e prendiamo solo i primi k utenti più simili.
    similar_users = [u for u in similar_users if u != user_index][:k]

    # 8. Troviamo i libri NON ancora valutati dall’utente target
    user_rated_items = mat[user_index].nonzero()[1]  # colonne valutate dall’utente target
    all_items = set(range(mat.shape[1]))             # tutti i libri
    items_to_predict = list(all_items - set(user_rated_items))

    # 9. Prediciamo i rating per ciascun libro non valutato
    preds = {}
    for item in items_to_predict:
        num, den = 0, 0
        for u in similar_users:
            # Se l’utente simile ha valutato questo libro
            if mat[u, item] != 0:
                # Numeratore = somma(similarità * (rating - media utente simile))
                num += similarities[u] * (mat[u, item] - user_means[u])
                # Denominatore = somma degli assoluti delle similarità
                den += abs(similarities[u])
        # Se abbiamo almeno un vicino che ha valutato l’item
        if den > 0:
            # Predizione = media utente target + (numeratore/denominatore)
            preds[item] = user_means[user_index] + num / den

    # 10. Ordiniamo i libri previsti per punteggio decrescente
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # 11. Convertiamo gli indici numerici degli item in ISBN originali
    inv_item_mapping = {v: k for k, v in item_mapping.items()}
    recommendations = [(inv_item_mapping[i], score) for i, score in top_items]

    return recommendations

# Esempio di utilizzo:
#ratings = pd.read_csv("ratings_explicit_mod.csv")
#
#recommendations = user_based_cf(user_id=276729, ratings=ratings, k=50, top_n=10)
#
#print("Raccomandazioni per l'utente 276729:")
#for isbn, score in recommendations:
#    print(isbn, score)
