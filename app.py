"""
Simple Python backend for SII recommender system project.
Provides basic endpoints for a recommender system.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
import pandas as pd

# Import service functions
from services import (
    get_book_info_by_isbn
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/getRecommenderSystemsInfos', methods=['GET'])
def get_recommender_systems_infos():
    # Placeholder for getting recommender systems information
    return jsonify({
            "recommender_systems": [
                {"id": 1, "name": "Top Popularity", "description": "Recommends the most popular items (Top 10 popular books)"},
                {"id": 2, "name": "User Based Collaborative Filtering", "description": "Recommends items based on user similarity (Users like you also liked...)"},
                {"id": 3, "name": "Item Based Collaborative Filtering", "description": "Recommends items based on item similarity (Who read these books also read...)"},
                {"id": 4, "name": "Content Based Filtering", "description": "Recommends items based on content features (Similar books based on description)"}
            ]
    })

@app.route('/getFiveRandomUsers', methods=['GET'])
def get_five_random_users():
    ratings = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
    five_random_users_ids = ratings['User-ID'].sample(n=5, random_state=1).tolist()
    users = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/users_mod.csv')
    random_users = users[users['User-ID'].isin(five_random_users_ids)].to_dict(orient='records')
    random_users_top_five_rated_books = []
    for user in random_users:
        user_id = user['User-ID']
        user_ratings = ratings[ratings['User-ID'] == user_id]
        top_five_rated_books = user_ratings.sort_values(by='Book-Rating', ascending=False).head(5)
        books_info = []
        for _, row in top_five_rated_books.iterrows():
            book_info = get_book_info_by_isbn(row['ISBN'])
            if book_info:
                book_info.update({'Book-Rating': row['Book-Rating']})
                books_info.append(book_info)
        random_users_top_five_rated_books.append({
            'User-ID': user_id,
            'Top-Five-Rated-Books': books_info
        })
        # Unite random_users and random_users_top_five_rated_books
    for user in random_users:
        for user_books in random_users_top_five_rated_books:
            if user['User-ID'] == user_books['User-ID']:
                user.update(user_books)

    return jsonify({ "random_users": random_users })

from scripts.top_popularity_rs import get_10_top_popular_books
@app.route('/get_top_popularity_rs', methods=['GET'])
def get_top_popularity_rs():
    """Get top 10 popular books based on ratings."""
    try:
        # Call service function to get top 10 popular books
        ratings_explicit_mod = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
        top_10_books = get_10_top_popular_books(ratings_explicit_mod)
        # Enrich with book details
        for book in top_10_books:
            book_info = get_book_info_by_isbn(book['ISBN'])
            if book_info:
                book.update(book_info)
        return jsonify({"top_10_books": top_10_books})
    except Exception as e:
        logger.error(f"Error getting top popularity recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve top popularity recommendations"
        }), 500
    
from scripts.user_based_cf_rs import user_based_cf
@app.route('/get_user_based_cf_rs', methods=['GET'])
def get_user_based_cf_rs():
    """Get user-based collaborative filtering recommendations."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({
            "status": "error",
            "message": "User ID is required"
        }), 400
    try:
        ratings_explicit_mod = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
        recommendations = user_based_cf(user_id, ratings=ratings_explicit_mod, k=50, top_n=10)
        top_10_books = []
        # Enrich with book details
        for isbn, score in recommendations:
            book_info = get_book_info_by_isbn(isbn)
            if book_info:
                book_info.update({'predicted_rating': score})
                top_10_books.append(book_info)
        return jsonify({"top_10_books": top_10_books})
    except Exception as e:
        logger.error(f"Error getting user-based CF recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve user-based CF recommendations"
        }), 500

from scripts.item_based_cf_rs import item_based_cf
@app.route('/get_item_based_cf_rs', methods=['GET'])
def get_item_based_cf_rs():
    """Get item-based collaborative filtering recommendations."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({
            "status": "error",
            "message": "User ID is required"
        }), 400
    try:
        ratings_explicit_mod = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset_mod/ratings_explicit_mod.csv')
        recommendations = item_based_cf(user_id, ratings=ratings_explicit_mod, k=50, top_n=10)
        top_10_books = []
        # Enrich with book details
        for isbn, score in recommendations:
            book_info = get_book_info_by_isbn(isbn)
            if book_info:
                book_info.update({'predicted_rating': score})
                top_10_books.append(book_info)
        return jsonify({"top_10_books": top_10_books})
    except Exception as e:
        logger.error(f"Error getting item-based CF recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve item-based CF recommendations"
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "SII Recommender System Backend is running",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "sii-recommender-backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


if __name__ == '__main__':
    logger.info("Starting SII Recommender System Backend...")
    app.run(host='0.0.0.0', port=5000, debug=True)