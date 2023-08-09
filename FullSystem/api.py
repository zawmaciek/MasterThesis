from typing import Any
from math import dist
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset import Dataset, movieId, userId, User
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

    def get_reccomendations_for_user(self, ratings: list[tuple[movieId, float]]) -> tuple[list[Any], Any]:
        user_vector = self.dataset.get_reduced_matrix_for_user(ratings)
        array = self.scaler.transform([user_vector])[0]
        array = self.pca.transform([array])[0]
        label = self.kmeans.predict([array])[0]
        users = self.get_users_from_label(label)
        movies = self.dataset.movies_from_users(users)
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

    def get_reccomendations_for_user(self, label: int, ratings: list[tuple[movieId, float]]) -> list[movieId]:
        users = self.dataset.get_user_full_vectors(label)
        vector_from_rankings = self.dataset.get_full_vector_from_ratings(ratings)
        ranked_users = [User(u.id, u.vector, self.get_distance(vector_from_rankings, u.vector)) for u in users]
        sorted_list = sorted(ranked_users, key=lambda x: x.rank, reverse=True)
        top = [a.id for a in sorted_list][:3]
        return self.dataset.movies_from_users(top, [a[0] for a in ratings])


class System:
    def __init__(self):
        self.dataset = Dataset()
        self.kmeans = S1Kmeans(self.dataset)
        self.dataset.label_by_user = self.kmeans.get_user_label_map()
        self.cf_user = S2CF(self.dataset)

    def get_recommendations_for_user(self, ratings: list[tuple[movieId, float]]) -> list[movieId]:
        rec1, label = self.kmeans.get_reccomendations_for_user(ratings)
        rec2 = self.cf_user.get_reccomendations_for_user(label, ratings)
        return rec2


s = System()
user = [(movieId(1219), 1.0), (movieId(1085), 1.0), (movieId(491), 1.0), (movieId(1269), 1.0)]
print(s.get_recommendations_for_user(user))
