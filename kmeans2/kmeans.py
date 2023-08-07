from collections import Counter, defaultdict
from typing import Tuple, List, Any

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset import Dataset, movieId, userId
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from validation import stereotypical_fans


class KMeansSystem():
    def __init__(self, clusters_num: int) -> None:
        self.matrix = None
        self.scaler = None
        self.labels = None
        self.kmeans = None
        self.pca = None
        self.dataset = Dataset()
        self.create_model(clusters_num)

    def sample_from_np(self, matrix, prcntg=0.5):
        number_of_rows = matrix.shape[0]
        random_indices = np.random.choice(number_of_rows,
                                          size=int(matrix.shape[0] // (1 // prcntg)),
                                          replace=False)
        # display random rows
        matrix = matrix[random_indices, :]
        return matrix

    def create_model(self, c_num: int) -> None:
        self.matrix = self.dataset.get_matrix()
        # SAMPLE RANDOM
        # self.matrix = self.sample_from_np(matrix, 0.1)
        self.scaler = StandardScaler()
        print("scaling")
        scaled_features = self.scaler.fit_transform(self.matrix)
        print("pca")
        self.pca = PCA(n_components=13)
        scaled_features = self.pca.fit_transform(scaled_features)
        print("training model")
        self.kmeans = KMeans(n_clusters=c_num, random_state=0, n_init="auto")
        self.labels = self.kmeans.fit(scaled_features)

    def infer(self, ratings: list[tuple[movieId, float]]) -> tuple[list[Any], Any]:
        user_vector = self.dataset.get_matrix_for_user(ratings)
        array = self.scaler.transform([user_vector])[0]
        array = self.pca.transform([array])[0]
        label = self.kmeans.predict([array])[0]
        print(label)
        users = self.get_user_from_label(label)
        print(f"count: {len(users)}")
        movies = self.movies_from_users(users)
        return movies, label

    def get_user_from_label(self, label: int) -> list[userId]:
        indices = np.where(self.labels.labels_ == label)[0]
        u_ids = [self.dataset.matrix_user_id_to_user_id[id] for id in indices]
        return u_ids

    def movies_from_users(self, users: list[userId]):
        ratings = self.dataset.ratings_by_user
        ddict = defaultdict(lambda: 0.0)
        movies_map = {a.movieId: a for a in self.dataset.all_movies}
        for user in users:
            for m_id, rating in ratings[user]:
                ddict[m_id] += rating
        top = sorted(ddict.items(), key=lambda item: item[1], reverse=True)[0:10]
        return [movies_map[a[0]].title for a in top]

    def get_vector_for_users(self, users: list[userId]):
        users_vector = [0 for _ in range(len(self.dataset.GENRES))]
        movies_map = {a.movieId: a for a in self.dataset.all_movies}
        for user in tqdm(users, desc="get matrix for users"):
            for m_id, rating in self.dataset.ratings_by_user[user]:
                if m_id in movies_map:
                    movie_vector = movies_map[m_id].genres_vector
                    users_vector = [users_vector[i] + movie_vector[i] * rating for i in range(len(self.dataset.GENRES))]
        s = sum(users_vector)
        users_vector = [i / s for i in users_vector]
        return {self.dataset.GENRES[i]: users_vector[i] for i in range(len(users_vector))}

    def add_group_for_users(self):
        for label in tqdm(range(50), desc="putting in"):
            users = self.get_user_from_label(label)
            for user in users:
                self.dataset.add_group_to_user(user, label)


k = KMeansSystem(50)
