import random
import time

import numpy as np
from typing import Any
from math import dist

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataset import Dataset, movieId, userId, User, TaggedMovie
from sklearn.decomposition import PCA


class S1Kmeans:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.kmeans = self.get_model()

    def get_model(self):
        self.matrix = self.dataset.get_reduced_matrix()
        self.scaler = StandardScaler()
        print("scaling")
        scaled_features = self.scaler.fit_transform(self.matrix)
        print("pca")
        self.pca = PCA(n_components=13)
        scaled_features = self.pca.fit_transform(scaled_features)
        print("training model")
        kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto")
        self.labels = kmeans.fit(scaled_features)
        print("TRAINING DONE")
        return kmeans

    def get_users_from_label(self, label: int) -> list[userId]:
        indices = np.where(self.labels.labels_ == label)[0]
        u_ids = [self.dataset.matrix_user_id_to_user_id[id] for id in indices]
        return u_ids

    def get_reccomendations_for_user(self, ratings: list[tuple[movieId, float]]) -> tuple[list[movieId], Any]:
        user_vector = self.dataset.get_reduced_matrix_for_user(ratings)
        array = self.scaler.transform([user_vector])[0]
        array = self.pca.transform([array])[0]
        label = self.kmeans.predict([array])[0]
        users = self.get_users_from_label(label)
        movies = self.dataset.get_movie_ids_from_users(users, [a[0] for a in ratings])
        return movies, label

    def get_user_label_map(self) -> dict[userId, int]:
        d = dict()
        for label in range(50):
            users = self.get_users_from_label(label)
            for user in users:
                d[user] = label
        return d


class S2CF:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.user_to_label_map = self.dataset.label_by_user

    @staticmethod
    def get_distance(u1: list[float], u2: list[float]) -> float:
        return dist(u1, u2)

    def get_reccomendations_for_user(self, label: int, ratings: list[tuple[movieId, float]], ignored: list[movieId]) -> list[movieId]:
        users = self.dataset.get_users_full_vectors(label)
        vector_from_rankings = self.dataset.get_full_user_vector_from_ratings(ratings)
        ranked_users = [User(u.id, u.vector, self.get_distance(vector_from_rankings, u.vector)) for u in users]
        sorted_list = sorted(ranked_users, key=lambda x: x.rank, reverse=True)
        top = [a.id for a in sorted_list if a.id not in ignored][:3]
        return self.dataset.get_movie_ids_from_users(top, [a[0] for a in ratings])


class S3CF:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.user_to_label_map = self.dataset.label_by_user
        self.all_movie_vectors = self.dataset.get_movies_vectors()

    @staticmethod
    def get_distance(m1: list[float], m2: list[float]) -> float:
        return dist(m1, m2)

    def get_reccomendations_for_movie(self, m_id: movieId, ignored: list[movieId]) -> list[movieId]:
        vector_from_rankings = self.dataset.movie_vector_mapping[m_id]
        ranked_users = [TaggedMovie(u.movieId, u.tags, self.dataset.movie_id_title_mapping[u.movieId], self.get_distance(vector_from_rankings, u.tags)) for u in
                        self.all_movie_vectors]
        sorted_list = sorted(ranked_users, key=lambda x: x.rank, reverse=True)
        top = [a.movieId for a in sorted_list if a.movieId not in ignored][:10]
        return top


class System:
    def __init__(self):
        self.dataset = Dataset()
        self.kmeans = S1Kmeans(self.dataset)
        self.dataset.label_by_user = self.kmeans.get_user_label_map()
        self.cf_user = S2CF(self.dataset)
        self.cf_movies = S3CF(self.dataset)

    def get_recommendations_for_user(self, ratings: list[tuple[movieId, float]]) -> tuple[DataFrame, DataFrame]:
        t = time.time()
        rec1, label = self.kmeans.get_reccomendations_for_user(ratings)
        rec1 = random.sample(rec1, 5)
        rec2 = self.cf_user.get_reccomendations_for_user(label, ratings, rec1)
        rec2 = random.sample(rec2, 5)
        rec3 = self.cf_movies.get_reccomendations_for_movie(ratings[0][0], rec1 + rec2)
        rec3 = random.sample(rec3, 5)
        ensemble = rec1[:2] + rec2[:2] + rec3[:2]
        print(f"LOADING TOOK {time.time() - t} seconds")
        return pd.DataFrame({"GENERAL": [self.dataset.movie_id_title_mapping[a] for a in rec1],
                             "SIMILAR": [self.dataset.movie_id_title_mapping[a] for a in rec2],
                             "SPECIFIC": [self.dataset.movie_id_title_mapping[a] for a in rec3]}), \
            pd.DataFrame({"ENSEMBLE": [self.dataset.movie_id_title_mapping[a] for a in ensemble]})


if __name__ == "__main__":
    s = System()
    stereotypical_fans = dict()
    stereotypical_fans["comedy"] = [(1219, 1.0), (1085, 1.0), (491, 1.0),
                                    (1269, 1.0)]  # blues brothers, dial m for murder,manhattan murder mystery, back to the future
    stereotypical_fans["children"] = [(33, 1.0), (593, 1.0), (594, 1.0), (1012, 1.0)]  # babe, snow white,beauty and the beast,parent trap
    stereotypical_fans["adventure"] = [(647, 1.0), (779, 1.0), (1100, 1.0), (1269, 1.0)]  # mission impossible, idenpenence day, top gun,back to the future
    for user in stereotypical_fans:
        print(user)
        user_movies = [(movieId(a[0]), a[1]) for a in stereotypical_fans[user]]
        s.get_recommendations_for_user(user_movies)
        print(s)
