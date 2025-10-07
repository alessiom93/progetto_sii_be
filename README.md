# progetto_sii_be
Backend in Python for SII recommender system university project

## Overview
This is a simple Python backend with basic endpoints for a recommender system. It provides RESTful APIs to manage and retrieve recommendations.

## Setup and Installation

# Run Python File
# conda activate machine_learning_001
# python c:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/scripts/run_user_vs_item.py


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

## API Endpoints - general

### 1. Health Check
- **GET** `/`
- Returns basic health status

### 2. API Health Check
- **GET** `/api/health`
- Returns detailed API health information

### 3. Get Recommender Systems Informations
- **GET** `/getRecommenderSystemsInfos`
- Returns all supported recommender systems

### 4. Get Five Random Users
- **GET** `/getFiveRandomUsers`
- Returns five random users

### API Endpoints - recommender systems

### 5. Get Top 10 Most Popular Books
- **GET** `/get_top_popularity_rs`
- Returns 10 books, the most popular ones

### 6. Get Top 10 Books With User Based Collaborative Filtering
- **GET** `/get_user_based_cf_rs`
- Returns 10 books, choosen by the algorithm

### 7. Get Top 10 Books With Item Based Collaborative Filtering
- **GET** `/get_item_based_cf_rs`
- Returns 10 books, choosen by the algorithm
