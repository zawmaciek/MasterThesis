import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset import Dataset, movieId
import numpy as np
import pandas as pd


class KMeansSystem():
    def __init__(self) -> None:
        self.labels = None
        self.kmeans = None
        self.dataset = Dataset()
        self.create_model()

    def create_model(self) -> None:
        matrix = self.dataset.get_matrix()
        # SAMPLE RANDOM
        number_of_rows = matrix.shape[0]
        random_indices = np.random.choice(number_of_rows,
                                          size=matrix.shape[1] // 2,
                                          replace=False)
        # display random rows
        matrix = matrix[random_indices, :]
        scaler = StandardScaler()
        print("scaling")
        scaled_features = scaler.fit_transform(matrix)
        print("training model")
        self.kmeans = KMeans(n_clusters=100, random_state=0, n_init="auto")
        self.labels = self.kmeans.fit(scaled_features)

    def infer(self, ratings:list[tuple[movieId, float]]):
        array = [0 for i in self.dataset.movies_count]
        for movie, rating in ratings:
            array[self.dataset.movie_id_to_matrix_movie_id[movie]] = rating
        label = self.kmeans.predict([array])[0]
        print(label)
        print(self.labels)
        return None


horror_fan = [(movieId(407), 1.0), (movieId(426), 1.0), (movieId(512), 1.0),
              (movieId(593), 1.0)]  # In the mouth of maddness, Body snatchers,puppet master, silence of the lamb
k = KMeansSystem()
k.infer(horror_fan)
