"""
Simple Python backend for SII recommender system project.
Provides basic endpoints for a recommender system.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

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

from scripts.top_popularity_rs import get_10_top_popular_books
@app.route('/get_top_popularity_rs', methods=['GET'])
def get_top_popularity_rs():
    """Get top 10 popular books based on ratings."""
    try:
        # Call service function to get top 10 popular books
        top_10_books = get_10_top_popular_books()
        # Enrich with book details
        for book in top_10_books:
            book_info = get_book_info_by_isbn(book['ISBN'])
            if book_info:
                book.update(book_info)
        return jsonify({
            "status": "success",
            "data": top_10_books
        })
    except Exception as e:
        logger.error(f"Error getting top popularity recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve top popularity recommendations"
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