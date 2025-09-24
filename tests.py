# Ignore warnings for better log readability
import warnings
warnings.filterwarnings('ignore')

# To be used as a breakpoint
"""
import sys
sys.exit()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import ipywidgets as widgets
#from IPython.display import display, clear_output
import os, sys
import re

# Dont truncate column width when displaying dataframes
# pd.set_option('display.max_colwidth', None)

books = pd.read_csv('./dataset/Books.csv')
# dropping last three columns containing image URLs which will not be required for analysis (axis=1 means column-wise operation) (inplace=True means modify the dataframe in place)
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)
print('Books dataset loaded with shape:', books.shape)
print('Head:', books.head(), '\n')

users = pd.read_csv('./dataset/Users.csv')
print('Users dataset loaded with shape:', users.shape)
print('Head:', users.head(), '\n')

ratings = pd.read_csv('./dataset/Ratings.csv')
print('Ratings dataset loaded with shape:', ratings.shape)
print('Head:', ratings.head(), '\n')


# EXPLORATORY DATA ANALYSIS (EDA).
# Checking for missing values.
print('Books missing values:\n', books.isnull().sum(), '\n')
print('Users missing values:\n', users.isnull().sum(), '\n')
print('Ratings missing values:\n', ratings.isnull().sum(), '\n')

# Checking for unique and possibly invalid values in Books dataset
# pd.set_option('display.max_colwidth', None)
print('Book unique values:')
for col in books.columns:
    print(f"Unique values in '{col}': {books[col].unique()}\n")
# Checking for unique and possibly invalid values in Users dataset
print('User unique values:')
for col in users.columns:
    print(f"Unique values in '{col}': {users[col].unique()}\n")
# Checking for unique and possibly invalid values in Ratings dataset
print('Rating unique values:')
for col in ratings.columns:
    print(f"Unique values in '{col}': {ratings[col].unique()}\n")


# Books dataset has two rows with 'DK Publishing Inc' as yearOfPublication
print('Books rows with invalid yearOfPublication values:')
print(books.loc[books.yearOfPublication == 'DK Publishing Inc',:], '\n')
# Books dataset has one row with 'Gallimard' as yearOfPublication
print('Books rows with invalid yearOfPublication values:')
print(books.loc[books.yearOfPublication == 'Gallimard',:], '\n')
# Books dataset has one row with '0' as yearOfPublication, show only the count of such rows
print('Books rows with 0 yearOfPublication values count:', books.loc[books.yearOfPublication == '0',:].shape[0], '\n')

# INCORRECT DATA CORRECTION
# BOOKS DATASET
print('Correcting data columns mismatch in Books dataset\n')
# Correcting bookAuthor -> yearOfPublication, yearOfPublication -> publisher mismatch and correcting bookTitle for the two rows with 'DK Publishing Inc'
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953', 'bookAuthor'] = 'James Buckley'
books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953', 'publisher'] = 'DK Publishing Inc'
books.loc[books.ISBN == '0789466953', 'bookTitle'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
#ISBN '078946697X'
books.loc[books.ISBN == '078946697X', 'bookAuthor'] = 'Michael Teitelbaum'
books.loc[books.ISBN == '078946697X', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X', 'publisher'] = 'DK Publishing Inc'
books.loc[books.ISBN == '078946697X', 'bookTitle'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
# Verifying the corrections
print('Verifying corrections:')
print(books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:])
# Correcting bookAuthor -> yearOfPublication, yearOfPublication -> publisher mismatch for the row with 'Gallimard'
#ISBN '2070426769'
books.loc[books.ISBN == '2070426769', 'bookAuthor'] = 'Jean-Marie Gustave'
books.loc[books.ISBN == '2070426769', 'yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769', 'publisher'] = 'Gallimard'
books.loc[books.ISBN == '2070426769', 'bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
# Verifying the correction
print('Verifying correction:')
print(books.loc[books.ISBN == '2070426769',:], '\n')
# Replacing yearOfPublication '0' with NaN
print('Replacing yearOfPublication 0 with NaN, rows count with 0:', books.loc[books.yearOfPublication == '0',:].shape[0])
books.loc[books.yearOfPublication == '0', 'yearOfPublication'] = np.NaN
# Verifying the replacement
print('Verifying replacement of 0 with NaN in yearOfPublication, rows count with NaN:', books.loc[books.yearOfPublication.isna(),:].shape[0])
# Removing rows with NaN yearOfPublication
books.dropna(subset=['yearOfPublication'], inplace=True)
# Verifying the replacement
print('Verifying removal of rows with NaN in yearOfPublication, rows count with NaN:', books.loc[books.yearOfPublication.isna(),:].shape[0])
# resetting the dtype as int32
books.yearOfPublication = books.yearOfPublication.astype(np.int32)
# Removing rows with bookAuthor missing values
print('Removing rows with bookAuthor missing values, rows count with NaN:', books.loc[books.bookAuthor.isna(),:].shape[0])
books.dropna(subset=['bookAuthor'], inplace=True)
# Verifying the removal
print('Verifying removal, rows count with NaN:', books.loc[books.bookAuthor.isna(),:].shape[0])
# Removing rows with publisher missing values
print('Removing rows with publisher missing values, rows count with NaN:', books.loc[books.publisher.isna(),:].shape[0])
books.dropna(subset=['publisher'], inplace=True)
# Verifying the removal
print('Verifying removal, rows count with NaN:', books.loc[books.publisher.isna(),:].shape[0])
print('Books missing values:\n', books.isnull().sum(), '\n')

# USERS DATASET
print('Correcting data columns in Users dataset\n')
# Finding incorrect values in 'age' column
print('Users rows with age values < 5:', users.loc[users.Age < 5, :].shape[0])
print('Users rows with age values > 100:', users.loc[users.Age > 100, :].shape[0])
print('Users rows with age values = NaN:', users.loc[users.Age.isna(), :].shape[0])
# Replacing age values < 5 or > 100 with NaN
print('Replacing age values < 5 or > 100 with NaN')
users.loc[(users.Age < 5) | (users.Age > 100), 'Age'] = np.NaN
# Verifying the replacement
print('Users rows with age values < 5:', users.loc[users.Age < 5, :].shape[0])
print('Users rows with age values > 100:', users.loc[users.Age > 100, :].shape[0])
print('Users rows with age values = NaN:', users.loc[users.Age.isna(), :].shape[0])
# Replacing NaN age values with median age
print('Replacing NaN age values with median age')
median_age = int(users.Age.median())
users.loc[users.Age.isna(), 'Age'] = median_age
# Verifying the replacement
print('Users rows with age values = NaN:', users.loc[users.Age.isna(), :].shape[0])
# resetting the dtype as int32
users.Age = users.Age.astype(np.int32)
# Finding incorrect values in 'Location' column
print('Users rows with invalid Location values (not matching city, state, country format), count:', users.loc[~users.Location.str.match(r'^[a-zA-Z\s]+,\s[a-zA-Z\s]+,\s[a-zA-Z\s]+$'), :].shape[0])
print('Users rows with location values = "n/a" in city, state or country, count:', users.loc[users.Location.str.contains(r'\bn/a\b', flags=re.IGNORECASE, regex=True), :].shape[0])
print('Users rows with location values = "none" in city, state or country, count:', users.loc[users.Location.str.contains(r'\bnone\b', flags=re.IGNORECASE, regex=True), :].shape[0])
print('Users rows with location values = "" in city, state or country, count:', users.loc[users.Location.str.contains(r'^\s*$', flags=re.IGNORECASE, regex=True), :].shape[0])
print('Users rows with location values = " " in city, state or country, count:', users.loc[users.Location.str.contains(r'^\s*$', flags=re.IGNORECASE, regex=True), :].shape[0])
# TODO: Further clean the Location column if required
print('Users missing values:\n', users.isnull().sum(), '\n')

# RATINGS DATASET
print('Correcting data columns in Ratings dataset\n')
# Finding incorrect values in 'bookRating' column
print('Ratings rows with bookRating values < 0, count:', ratings.loc[ratings['Book-Rating'] < 0, :].shape[0])
print('Ratings rows with bookRating values > 10, count:', ratings.loc[ratings['Book-Rating'] > 10, :].shape[0])
print('Ratings rows with duplicate ratings by same user for same book, count:', ratings.loc[ratings.duplicated(subset=['User-ID', 'ISBN'], keep=False), :].shape[0])
# Check if there are ISBNs in Ratings dataset which are not present in Books dataset
print('Ratings rows with ISBNs not present in Books dataset, count:', ratings.loc[~ratings.ISBN.isin(books.ISBN), :].shape[0])
# Check if there are User-IDs in Ratings dataset which are not present in Users dataset
print('Ratings rows with User-IDs not present in Users dataset, count:', ratings.loc[~ratings['User-ID'].isin(users['User-ID']), :].shape[0])
# Removing rows with ISBNs not present in Books dataset
print('Removing rows with ISBNs not present in Books dataset')
ratings = ratings[ratings.ISBN.isin(books.ISBN)]
print('Verifying: Ratings rows with ISBNs not present in Books dataset, count:', ratings.loc[~ratings.ISBN.isin(books.ISBN), :].shape[0])
print('Ratings missing values:\n', ratings.isnull().sum(), '\n')
# Checking ecplicit vs implicit ratings
print('Explicit ratings (1-10) count:', ratings.loc[ratings['Book-Rating'] != 0, :].shape[0])
print('Implicit ratings (0) count:', ratings.loc[ratings['Book-Rating'] == 0, :].shape[0], '\n')

