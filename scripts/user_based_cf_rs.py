import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Costruisce la matrice utente × item (sparse) dai rating.
    
    Ogni riga = un utente
    Ogni colonna = un libro
    Cella = rating dato dall'utente (0 se non valutato)
    
    Ritorna:
        mat : csr_matrix
            Matrice sparsa utente × item
        user_mapping : dict
            Indice numerico → User-ID
        item_mapping : dict
            Indice numerico → ISBN
    """
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    
    user_ids = ratings['User-ID'].astype('category')
    item_ids = ratings['ISBN'].astype('category')
    
    user_mapping = dict(enumerate(user_ids.cat.categories))
    item_mapping = dict(enumerate(item_ids.cat.categories))
    
    mat = csr_matrix(
        (ratings['Book-Rating'].astype(float),
         (user_ids.cat.codes, item_ids.cat.codes))
    )
    
    return mat, user_mapping, item_mapping

def user_based_cf(user_id, ratings, k=50, top_n=10, min_rating=0, max_rating=10):
    """
    Raccomandazioni User-based CF con Pearson correlation (tramite cosine similarity su matrice centrata).
    
    Parametri:
        user_id : int o str
            ID utente target
        ratings : pd.DataFrame
            Colonne ['User-ID', 'ISBN', 'Book-Rating']
        k : int
            Numero di vicini simili da considerare
        top_n : int
            Numero di libri da raccomandare
        min_rating, max_rating : float
            Limiti per il clipping dei rating predetti
    
    Ritorna:
        recommendations : list of tuples
            Lista (ISBN, rating previsto) ordinata per rating decrescente
    """
    # 1️⃣ Matrice utente × item
    mat, user_mapping, item_mapping = build_user_item_matrix(ratings)
    
    # 2️⃣ Controllo utente
    user_id = str(user_id)
    if user_id not in user_mapping.values():
        raise ValueError(f"User ID {user_id} non presente nel dataset")
    user_index = np.where(np.array(list(user_mapping.values())) == user_id)[0][0]
    
    # 3️⃣ Media rating utenti
    user_means = np.array(mat.sum(axis=1)).flatten() / (mat != 0).sum(axis=1).A1
    
    # 4️⃣ Centriamo la matrice rispetto alla media utente
    mat_centered = mat.copy().astype(float)
    for u in range(mat.shape[0]):
        start, end = mat_centered.indptr[u], mat_centered.indptr[u+1]
        if start < end:
            mat_centered.data[start:end] -= user_means[u]
    
    # 5️⃣ Similarità coseno tra utente target e tutti gli altri
    similarities = cosine_similarity(mat_centered[user_index], mat_centered).flatten()
    
    # 6️⃣ Top k utenti simili, escludendo se stesso
    similar_users = np.argsort(similarities)[::-1]
    similar_users = similar_users[similar_users != user_index][:k]
    
    # 7️⃣ Libri non ancora valutati dall'utente target
    user_rated_items = mat[user_index].nonzero()[1]
    all_items = set(range(mat.shape[1]))
    items_to_predict = list(all_items - set(user_rated_items))
    
    # 8️⃣ Predizione rating
    preds = {}
    for item in items_to_predict:
        num, den = 0.0, 0.0
        for u in similar_users:
            rating = mat[u, item]
            if rating != 0:
                sim = float(similarities[u])
                num += sim * (rating - user_means[u])
                den += abs(sim)
        if den > 0:
            pred = user_means[user_index] + num / den
            # Clipping dei rating
            pred = max(min_rating, min(max_rating, pred))
            preds[item] = pred
    
    # 9️⃣ Ordino e prendo top N
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 1️⃣0️⃣ Map indices → ISBN
    recommendations = [(item_mapping[i], score) for i, score in top_items]
    
    return recommendations
