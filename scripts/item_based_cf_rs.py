import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_item_user_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice item × user (sparse) dai rating.
    
    Ogni riga = un libro
    Ogni colonna = un utente
    Cella = rating dato dall'utente (0 se non valutato)
    
    Ritorna:
        mat : csr_matrix
            Matrice sparsa item × user
        item_mapping : dict
            Indice numerico → ISBN
        user_mapping : dict
            Indice numerico → User-ID
    """
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    
    user_ids = ratings['User-ID'].astype('category')
    item_ids = ratings['ISBN'].astype('category')
    
    item_mapping = dict(enumerate(item_ids.cat.categories))
    user_mapping = dict(enumerate(user_ids.cat.categories))
    
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (item_ids.cat.codes, user_ids.cat.codes))
    )
    
    return mat, item_mapping, user_mapping

def item_based_cf(user_id, ratings, k=50, top_n=10, min_rating=0, max_rating=10):
    """
    Raccomandazioni Item-based CF con Pearson correlation
    (tramite cosine similarity su matrice centrata per libro).
    
    Parametri:
        user_id : int o str
            ID utente target
        ratings : pd.DataFrame
            Colonne ['User-ID', 'ISBN', 'Book-Rating']
        k : int
            Numero di libri simili da considerare
        top_n : int
            Numero di libri da raccomandare
        min_rating, max_rating : float
            Limiti per clipping dei rating predetti
    
    Ritorna:
        recommendations : list of tuples
            Lista (ISBN, rating previsto) ordinata per rating decrescente
    """
    # 1️⃣ Matrice item × user
    mat, item_mapping, user_mapping = build_item_user_matrix(ratings)
    
    # 2️⃣ Controllo utente
    user_id = str(user_id)
    if user_id not in user_mapping.values():
        raise ValueError(f"User ID {user_id} non presente nel dataset")
    user_index = np.where(np.array(list(user_mapping.values())) == user_id)[0][0]
    
    # 3️⃣ Media rating per libro (centratura)
    item_means = np.array(mat.sum(axis=1)).flatten() / (mat != 0).sum(axis=1).A1
    
    # 4️⃣ Centriamo solo righe con rating non-zero
    mat_centered = mat.copy().astype(float)
    for i in range(mat.shape[0]):
        start, end = mat_centered.indptr[i], mat_centered.indptr[i+1]
        if start < end:
            mat_centered.data[start:end] -= item_means[i]
    
    # 5️⃣ Libri già valutati dall'utente target
    user_rated_items = mat[:, user_index].nonzero()[0]
    all_items = set(range(mat.shape[0]))
    items_to_predict = list(all_items - set(user_rated_items))
    
    # 6️⃣ Predizione dei rating per libri non valutati
    preds = {}
    for item in items_to_predict:
        # Trovo tutti i libri che l’utente ha già valutato
        rated_books = user_rated_items
        # Calcolo similarità coseno tra il libro target e tutti i libri valutati dall’utente
        sim = cosine_similarity(mat_centered[item], mat_centered[rated_books]).flatten()
        
        # Numeratore e denominatore per media pesata
        num, den = 0.0, 0.0
        for j, s in zip(rated_books, sim):
            rating = mat[j, user_index]
            if rating != 0:
                num += s * (rating - item_means[j])
                den += abs(s)
        if den > 0:
            pred = item_means[item] + num / den
            pred = max(min_rating, min(max_rating, pred))
            preds[item] = pred
    
    # 7️⃣ Ordino e prendo top N
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 8️⃣ Mappa indici numerici → ISBN
    recommendations = [(item_mapping[i], score) for i, score in top_items]
    
    return recommendations
