"""
Service layer for the SII Recommender System.
Contains business logic for recommendation operations.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def get_book_info_by_isbn(isbn: str) -> Optional[Dict]:
    try:
        books_df = pd.read_csv('C:/Users/alemo/OneDrive/Lavoro/progetto_sii_be/dataset/Books.csv')
        book_info = books_df[books_df['ISBN'] == isbn]
        if not book_info.empty:
            return book_info.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error retrieving book info for ISBN {isbn}: {str(e)}")
    return None
    b