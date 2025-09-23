"""
Simple Python backend for SII recommender system project.
Provides basic endpoints for a recommender system.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# In-memory storage for recommendations (in a real app, this would be a database)
recommendations_db = [
    {
        "id": 1,
        "user_id": "user1",
        "item": "Product A",
        "score": 0.95,
        "created_at": "2024-01-01T10:00:00Z"
    },
    {
        "id": 2,
        "user_id": "user1", 
        "item": "Product B",
        "score": 0.87,
        "created_at": "2024-01-01T10:05:00Z"
    }
]

# Counter for generating new IDs
next_id = 3


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
            filtered_recommendations = [
                rec for rec in recommendations_db 
                if rec['user_id'] == user_id
            ]
            return jsonify({
                "status": "success",
                "data": filtered_recommendations,
                "count": len(filtered_recommendations),
                "user_id": user_id
            })
        else:
            # Return all recommendations
            return jsonify({
                "status": "success",
                "data": recommendations_db,
                "count": len(recommendations_db)
            })
            
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve recommendations"
        }), 500


@app.route('/api/recommendations', methods=['POST'])
def create_recommendation():
    """Create a new recommendation."""
    try:
        global next_id
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
            
        if not all(key in data for key in ['user_id', 'item', 'score']):
            return jsonify({
                "status": "error",
                "message": "Missing required fields: user_id, item, score"
            }), 400
        
        # Validate score range
        if not (0 <= data['score'] <= 1):
            return jsonify({
                "status": "error",
                "message": "Score must be between 0 and 1"
            }), 400
        
        # Create new recommendation
        new_recommendation = {
            "id": next_id,
            "user_id": data['user_id'],
            "item": data['item'],
            "score": float(data['score']),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        recommendations_db.append(new_recommendation)
        next_id += 1
        
        logger.info(f"Created new recommendation with ID {new_recommendation['id']}")
        
        return jsonify({
            "status": "success",
            "message": "Recommendation created successfully",
            "data": new_recommendation
        }), 201
        
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