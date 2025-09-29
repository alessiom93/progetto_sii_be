# Item-based collaborative filtering recommendation system
# Pearson correlation for items similarity [-1, +1]
# Input: ratings-matrix and user-id
# Find similar items by users ratings
# Calculate predictions for items not rated by the user, based on ratings of similar items by the user
# Output: list of recommended books (ISBNs)

# Chi ha letto questo libro ha anche letto...

import pandas as pd

def item_based_cf(user_id):
    ratings_explicit_mod = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
    # Find similar items
    similar_items = find_similar_items(ratings_explicit_mod, user_id)
    top_similar_items = similar_items[:100]  # Top 100 similar items with positive similarity
    # Predict ratings for books not yet rated by the user
    user_rated_books = set(get_books_rated_by_user(user_id, ratings_explicit_mod))
    all_books = set(ratings_explicit_mod['ISBN'].unique())
    books_to_predict = all_books - user_rated_books
    predicted_ratings = calculate_predictions(user_id, books_to_predict, top_similar_items, ratings_explicit_mod)
    return predicted_ratings[:10]  # Return top 10 recommendations

def find_similar_items(ratings_matrix, user_id):
    user_rated_books = get_books_rated_by_user(user_id, ratings_matrix)
    all_books = ratings_matrix['ISBN'].unique()
    all_books = [b for b in all_books if b not in user_rated_books]
    similarities = []
    for other_book in all_books:
        for user_book in user_rated_books:
            similarity = calculate_item_similarity(other_book, user_book, ratings_matrix)
            similarities.append((other_book, similarity))
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Remove negative similarities
    similarities = [(b, sim) for b, sim in similarities if sim > 0]
    return similarities

def calculate_item_similarity(item1_isbn, item2_isbn, ratings_explicit_mod):
    # Find users who rated both items
    users_who_rated_both = get_users_who_rated_books([item1_isbn, item2_isbn], ratings_explicit_mod)
    if not users_who_rated_both:
        return 0  # No common raters, similarity is 0
    item1_avg_rating = calculate_item_average_rating(item1_isbn, ratings_explicit_mod)
    item2_avg_rating = calculate_item_average_rating(item2_isbn, ratings_explicit_mod)
    item_similarity_numerator = calculate_item_similarity_numerator(item1_isbn, item2_isbn, users_who_rated_both, item1_avg_rating, item2_avg_rating, ratings_explicit_mod)
    item_similarity_denominator = calculate_item_similarity_denominator(item1_isbn, item2_isbn, users_who_rated_both, item1_avg_rating, item2_avg_rating, ratings_explicit_mod)
    if item_similarity_denominator == 0:
        return 0
    return item_similarity_numerator / item_similarity_denominator

def get_books_rated_by_user(user_id, ratings_explicit_mod):
    user_ratings = ratings_explicit_mod[ratings_explicit_mod['User-ID'] == user_id]
    return user_ratings['ISBN'].tolist()

def get_users_who_rated_books(book_isbns, ratings_explicit_mod):
    users = ratings_explicit_mod[ratings_explicit_mod['ISBN'].isin(book_isbns)]['User-ID'].unique()
    return users.tolist()

def get_users_who_rated_book(book_isbn, ratings_explicit_mod):
    users = ratings_explicit_mod[ratings_explicit_mod['ISBN'] == book_isbn]['User-ID'].unique()
    return users.tolist()

def get_user_rating(user_id, book_isbn, ratings_explicit_mod):
    user_rating = ratings_explicit_mod[(ratings_explicit_mod['User-ID'] == user_id) & (ratings_explicit_mod['ISBN'] == book_isbn)]
    if not user_rating.empty:
        return user_rating.iloc[0]['Book-Rating']
    return None

def calculate_user_average_rating(user_id, ratings_explicit_mod):
    user_ratings = ratings_explicit_mod[ratings_explicit_mod['User-ID'] == user_id]
    if not user_ratings.empty:
        return user_ratings['Book-Rating'].mean()
    return 0

def calculate_item_average_rating(item_isbn, ratings_explicit_mod):
    item_ratings = ratings_explicit_mod[ratings_explicit_mod['ISBN'] == item_isbn]
    if not item_ratings.empty:
        return item_ratings['Book-Rating'].mean()
    return 0

def calculate_item_similarity_numerator(item1_isbn, item2_isbn, users_who_rated_both, item1_avg_rating, item2_avg_rating, ratings_explicit_mod):
    numerator = 0
    for user_id in users_who_rated_both:
        rating_item1 = get_user_rating(user_id, item1_isbn, ratings_explicit_mod)
        rating_item2 = get_user_rating(user_id, item2_isbn, ratings_explicit_mod)
        numerator += (rating_item1 - item1_avg_rating) * (rating_item2 - item2_avg_rating)
    return numerator

def calculate_item_similarity_denominator(item1_isbn, item2_isbn, users_who_rated_both, item1_avg_rating, item2_avg_rating, ratings_explicit_mod):
    sum1 = sum((get_user_rating(user_id, item1_isbn, ratings_explicit_mod) - item1_avg_rating) ** 2 for user_id in users_who_rated_both)
    sum2 = sum((get_user_rating(user_id, item2_isbn, ratings_explicit_mod) - item2_avg_rating) ** 2 for user_id in users_who_rated_both)
    return (sum1 ** 0.5) * (sum2 ** 0.5)

def calculate_prediction(user_id, isbn, similar_items, ratings_explicit_mod):
    numerator = 0
    denominator = 0
    for other_item_isbn, similarity in similar_items:
        other_item_rating = get_user_rating(user_id, other_item_isbn, ratings_explicit_mod)
        if other_item_rating is not None:
            numerator += similarity * (other_item_rating - calculate_item_average_rating(other_item_isbn, ratings_explicit_mod))
            denominator += abs(similarity) # Use absolute similarity to avoid negative influence
    if denominator == 0:
        return calculate_user_average_rating(user_id, ratings_explicit_mod)
    return calculate_item_average_rating(isbn, ratings_explicit_mod) + (numerator / denominator)

def calculate_predictions(user_id, books_to_predict, similar_items, ratings_explicit_mod):
    predictions = []
    for book in books_to_predict:
        predicted_rating = calculate_prediction(user_id, book, similar_items, ratings_explicit_mod)
        predictions.append((book, predicted_rating))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions