"""
Service layer for the SII Recommender System.
Contains business logic for recommendation operations.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

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


def get_all_recommendations() -> List[Dict]:
    """
    Get all recommendations from the database.
    
    Returns:
        List[Dict]: List of all recommendations
    """
    return recommendations_db


def get_recommendations_by_user(user_id: str) -> List[Dict]:
    """
    Get recommendations filtered by user_id.
    
    Args:
        user_id (str): The user ID to filter recommendations
        
    Returns:
        List[Dict]: List of recommendations for the specified user
    """
    return [
        rec for rec in recommendations_db 
        if rec['user_id'] == user_id
    ]


def validate_recommendation_data(data: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate recommendation data.
    
    Args:
        data (Dict): The recommendation data to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"
        
    required_fields = ['user_id', 'item', 'score']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate score range
    try:
        score = float(data['score'])
        if not (0 <= score <= 1):
            return False, "Score must be between 0 and 1"
    except (ValueError, TypeError):
        return False, "Score must be a valid number"
    
    return True, None


def create_recommendation(data: Dict) -> Dict:
    """
    Create a new recommendation.
    
    Args:
        data (Dict): The recommendation data
        
    Returns:
        Dict: The created recommendation
        
    Raises:
        ValueError: If the data is invalid
    """
    global next_id
    
    # Validate data
    is_valid, error_message = validate_recommendation_data(data)
    if not is_valid:
        raise ValueError(error_message)
    
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
    
    return new_recommendation


def get_recommendation_by_id(recommendation_id: int) -> Optional[Dict]:
    """
    Get a specific recommendation by ID.
    
    Args:
        recommendation_id (int): The ID of the recommendation
        
    Returns:
        Optional[Dict]: The recommendation if found, None otherwise
    """
    for rec in recommendations_db:
        if rec['id'] == recommendation_id:
            return rec
    return None


def delete_recommendation(recommendation_id: int) -> bool:
    """
    Delete a recommendation by ID.
    
    Args:
        recommendation_id (int): The ID of the recommendation to delete
        
    Returns:
        bool: True if deleted successfully, False if not found
    """
    global recommendations_db
    initial_length = len(recommendations_db)
    recommendations_db = [rec for rec in recommendations_db if rec['id'] != recommendation_id]
    
    if len(recommendations_db) < initial_length:
        logger.info(f"Deleted recommendation with ID {recommendation_id}")
        return True
    return False


def update_recommendation(recommendation_id: int, data: Dict) -> Optional[Dict]:
    """
    Update an existing recommendation.
    
    Args:
        recommendation_id (int): The ID of the recommendation to update
        data (Dict): The updated recommendation data
        
    Returns:
        Optional[Dict]: The updated recommendation if found, None otherwise
        
    Raises:
        ValueError: If the data is invalid
    """
    # Find the recommendation
    recommendation = get_recommendation_by_id(recommendation_id)
    if not recommendation:
        return None
    
    # Validate new data
    is_valid, error_message = validate_recommendation_data(data)
    if not is_valid:
        raise ValueError(error_message)
    
    # Update the recommendation
    recommendation['user_id'] = data['user_id']
    recommendation['item'] = data['item']
    recommendation['score'] = float(data['score'])
    recommendation['updated_at'] = datetime.utcnow().isoformat() + "Z"
    
    logger.info(f"Updated recommendation with ID {recommendation_id}")
    
    return recommendation


def get_recommendations_count() -> int:
    """
    Get the total count of recommendations.
    
    Returns:
        int: Total number of recommendations
    """
    return len(recommendations_db)


def get_recommendations_by_score_range(min_score: float, max_score: float) -> List[Dict]:
    """
    Get recommendations within a specific score range.
    
    Args:
        min_score (float): Minimum score threshold
        max_score (float): Maximum score threshold
        
    Returns:
        List[Dict]: List of recommendations within the score range
    """
    return [
        rec for rec in recommendations_db 
        if min_score <= rec['score'] <= max_score
    ]