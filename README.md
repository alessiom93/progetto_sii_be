# progetto_sii_be
Backend in Python for SII recommender system university project

## Overview
This is a simple Python backend with basic endpoints for a recommender system. It provides RESTful APIs to manage and retrieve recommendations.

## Features
- Health check endpoints
- Get recommendations (with optional user filtering)
- Create new recommendations
- JSON responses with proper error handling
- CORS support for frontend integration

## Setup and Installation

# Run Python File
# conda activate machine_learning_001
# python c:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/tests.py


### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
python app.py
```
Or use the provided script:
```bash
./run.sh
```

The server will start on http://localhost:5000

## API Endpoints

### 1. Health Check
- **GET** `/`
- Returns basic health status

### 2. API Health Check
- **GET** `/api/health`
- Returns detailed API health information

### 3. Get Recommendations
- **GET** `/api/recommendations`
- **GET** `/api/recommendations?user_id={user_id}`
- Returns all recommendations or filtered by user_id

### 4. Create Recommendation
- **POST** `/api/recommendations`
- Content-Type: application/json
- Body:
  ```json
  {
    "user_id": "string",
    "item": "string", 
    "score": number (0-1)
  }
  ```

## Example Usage

### Get all recommendations:
```bash
curl http://localhost:5000/api/recommendations
```

### Get recommendations for a specific user:
```bash
curl "http://localhost:5000/api/recommendations?user_id=user1"
```

### Create a new recommendation:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "item": "Product X", "score": 0.85}' \
  http://localhost:5000/api/recommendations
```

## Development
This is a simple implementation using Flask with in-memory storage. For production use, consider:
- Adding a proper database (PostgreSQL, MongoDB, etc.)
- Implementing authentication and authorization
- Adding input validation and sanitization
- Using a production WSGI server (Gunicorn, uWSGI)
- Adding logging and monitoring
