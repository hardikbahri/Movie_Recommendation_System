# Movie Recommendation System

This project is a movie recommendation system that suggests movies based on the user's input. It uses text-based features and various natural language processing (NLP) techniques to determine the similarity between movies.

## Features

- **User Input:** Accepts a movie name from the user.
- **Close Match:** Finds the closest match to the user's input from the dataset.
- **Similarity Score:** Computes similarity scores between movies using TF-IDF and cosine similarity.
- **Recommendations:** Recommends a list of similar movies based on the computed similarity scores.

## Concepts Used

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus). It is used for feature extraction from text data.

- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Measures how important a term is by comparing the number of documents containing the term to the total number of documents.

TF-IDF is calculated as:
\[ \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t) \]

### 2. Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors in a multidimensional space. It is used to determine how similar two documents (or movies, in this case) are based on their TF-IDF vectors.

Cosine similarity is calculated as:
\[ \text{cosine similarity} = \frac{A \cdot B}{\|A\| \|B\|} \]
where \(A\) and \(B\) are TF-IDF vectors.

### 3. Difflib
Difflib is a Python module that provides classes and functions for comparing sequences. It is used to find the closest match to the user's input movie name from the list of movie titles in the dataset.

## Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install the required libraries:**
   ```sh
   pip install numpy pandas scikit-learn
   ```

3. **Ensure you have the `movies.csv` file in the same directory as the script.**

## Usage

1. **Run the script:**
   ```sh
   python movie_recommendation_system.py
   ```

2. **Enter your favorite movie name:**
   ```
   Enter your favourite movie name: Iron Man
   ```

3. **Get recommendations:**
   ```
   Movies suggested for you: 
   1 . Iron Man
   2 . Iron Man 2
   3 . Iron Man 3
   ...
   ```

## Code Overview

### Loading Data

```python
# Loading the data from the CSV file to a pandas dataframe
movies_data = pd.read_csv('./movies.csv')
```

### Calculating Similarity

```python
# Getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
```

### User Input and Close Match

```python
# Getting the movie name from the user
movie_name = input(' Enter your favourite movie name : ')

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()

# Finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
```

### Finding Similar Movies

```python
# Finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# Getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))

# Sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
```

### Printing Recommendations

```python
# Print the name of similar movies based on the index
print('Movies suggested for you : \n')

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i < 30):
        print(i, '.', title_from_index)
        i += 1
```

