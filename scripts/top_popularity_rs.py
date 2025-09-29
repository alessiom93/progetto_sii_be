# Ignore warnings for better log readability
import warnings
warnings.filterwarnings('ignore')

# To be used as a breakpoint
"""
import sys
sys.exit()
"""

import pandas as pd

def get_10_top_popular_books(ratings_explicit_mod):
  # Sum the ratings for each book (ISBN) to determine popularity
  ratings_count = pd.DataFrame(ratings_explicit_mod.groupby(['ISBN'], as_index=False)['Book-Rating'].sum())
  # Get the top 10 most popular books based on summed ratings
  top_10 = ratings_count.sort_values('Book-Rating', ascending=False).head(10)
  return top_10.to_dict(orient='records')
