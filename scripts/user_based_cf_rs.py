# User-based collaborative filtering recommendation system
# Pearson correlation for user similarity [-1, +1]
# Input: ratings-matrix and user-id
# Find similar users based on rating patterns
# Aggregate their preferences to recommend books
# Output: list of recommended books (ISBNs)

import pandas as pd

def user_based_cf(ratings_matrix, user_id):
    ratings_explicit_mod = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
    # Find similar users
    similar_users = find_similar_users(ratings_explicit_mod, user_id)
    top_similar_users = similar_users[:100]  # Top 100 similar users with positive similarity
    # Predict ratings for books not yet rated by the user
    user_rated_books = set(get_books_rated_by_user(user_id, ratings_explicit_mod))
    all_books = set(ratings_explicit_mod['ISBN'].unique())
    books_to_predict = all_books - user_rated_books
    recommended_books = []
    for book in books_to_predict:
        predicted_rating = calculate_prediction(user_id, book, top_similar_users, ratings_explicit_mod)
        if predicted_rating > 0:
            recommended_books.append((book, predicted_rating))
    # Sort recommended books by predicted rating
    recommended_books.sort(key=lambda x: x[1], reverse=True)
    return recommended_books[:10]  # Return top 10 recommendations

def find_similar_users(ratings_matrix, user_id):
    all_users_ids_but_current = ratings_matrix.index[ratings_matrix.index != user_id]
    similarities = []
    for other_user_id in all_users_ids_but_current:
        similarity = calculate_users_similarity(user_id, other_user_id, ratings_matrix)
        similarities.append((other_user_id, similarity))
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def calculate_users_similarity(user1_id, user2_id, ratings_explicit_mod):
    # Find items rated by both users
    common_rated_books = get_books_rated_by_both_users(user1_id, user2_id, ratings_explicit_mod)
    # For each common item, sum the product of their ratings minus their average ratings
    if not common_rated_books:
        return 0  # No common ratings, similarity is 0
    user1_avg_rating = calculate_user_average_rating(user1_id, ratings_explicit_mod)
    user2_avg_rating = calculate_user_average_rating(user2_id, ratings_explicit_mod)
    pearson_numerator = calculate_pearson_numerator(user1_id, user2_id, common_rated_books, user1_avg_rating, user2_avg_rating, ratings_explicit_mod)
    pearson_denominator = calculate_pearson_denominator(user1_id, user2_id, common_rated_books, user1_avg_rating, user2_avg_rating, ratings_explicit_mod)
    # Compute Pearson correlation
    return pearson_numerator / pearson_denominator if pearson_denominator != 0 else 0

def get_books_rated_by_both_users(user1_id, user2_id, ratings_explicit_mod):
    books_rated_by_user1 = set(get_books_rated_by_user(user1_id, ratings_explicit_mod))
    books_rated_by_user2 = set(get_books_rated_by_user(user2_id, ratings_explicit_mod))
    return books_rated_by_user1.intersection(books_rated_by_user2)

def get_books_rated_by_user(user_id, ratings_explicit_mod):
    user_ratings = ratings_explicit_mod[ratings_explicit_mod['User-ID'] == user_id]
    return user_ratings['ISBN'].tolist()

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

def calculate_pearson_numerator(user1_id, user2_id, common_rated_books, user1_avg_rating, user2_avg_rating, ratings_explicit_mod):
    pearson_numerator = 0
    for book in common_rated_books:
        pearson_numerator += (get_user_rating(user1_id, book, ratings_explicit_mod) - user1_avg_rating) * (get_user_rating(user2_id, book, ratings_explicit_mod) - user2_avg_rating)
    return pearson_numerator

def calculate_pearson_denominator(user1_id, user2_id, common_rated_books, user1_avg_rating, user2_avg_rating, ratings_explicit_mod):
    pearson_denominator_user1 = 0
    for book in common_rated_books:
        pearson_denominator_user1 += (get_user_rating(user1_id, book, ratings_explicit_mod) - user1_avg_rating) ** 2
    pearson_denominator_user2 = 0
    for book in common_rated_books:
        pearson_denominator_user2 += (get_user_rating(user2_id, book, ratings_explicit_mod) - user2_avg_rating) ** 2
    return (pearson_denominator_user1 ** 0.5) * (pearson_denominator_user2 ** 0.5)

def calculate_prediction(user_id, isbn, similar_users, ratings_explicit_mod):
    numerator = 0
    denominator = 0
    for other_user_id, similarity in similar_users:
        other_user_rating = get_user_rating(other_user_id, isbn, ratings_explicit_mod)
        if other_user_rating is not None:
            numerator += similarity * (other_user_rating - calculate_user_average_rating(other_user_id, ratings_explicit_mod))
            denominator += abs(similarity)
    user_avg_rating = calculate_user_average_rating(user_id, ratings_explicit_mod)
    return user_avg_rating + (numerator / denominator if denominator != 0 else 0)