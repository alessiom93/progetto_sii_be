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
    get_all_recommendations,
    get_recommendations_by_user,
    create_recommendation,
    get_recommendations_count
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


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


@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get recommendations, optionally filtered by user_id."""
    try:
        user_id = request.args.get('user_id')
        
        if user_id:
            # Filter recommendations by user_id
            filtered_recommendations = get_recommendations_by_user(user_id)
            return jsonify({
                "status": "success",
                "data": filtered_recommendations,
                "count": len(filtered_recommendations),
                "user_id": user_id
            })
        else:
            # Return all recommendations
            all_recommendations = get_all_recommendations()
            return jsonify({
                "status": "success",
                "data": all_recommendations,
                "count": len(all_recommendations)
            })
            
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve recommendations"
        }), 500


@app.route('/api/recommendations', methods=['POST'])
def create_recommendation_endpoint():
    """Create a new recommendation."""
    try:
        data = request.get_json()
        
        # Create new recommendation using service function
        new_recommendation = create_recommendation(data)
        
        return jsonify({
            "status": "success",
            "message": "Recommendation created successfully",
            "data": new_recommendation
        }), 201
        
    except ValueError as e:
        # Handle validation errors
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Error creating recommendation: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to create recommendation"
        }), 500


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