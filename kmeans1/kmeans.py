from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset import Dataset, movieId
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CLUSTERS_NUM = 150


class KMeansSystem():
    def __init__(self) -> None:
        self.matrix = None
        self.scaler = None
        self.labels = None
        self.kmeans = None
        self.pca = None
        self.dataset = Dataset()
        self.create_model()

    def create_model(self) -> None:
        matrix = self.dataset.get_matrix()
        # SAMPLE RANDOM
        number_of_rows = matrix.shape[0]
        random_indices = np.random.choice(number_of_rows,
                                          size=matrix.shape[1] // 100,
                                          replace=False)
        # display random rows
        self.matrix = matrix[random_indices, :]
        self.scaler = StandardScaler()
        print("scaling")
        scaled_features = self.scaler.fit_transform(self.matrix)
        print("pca")
        self.pca = PCA(n_components=250)
        pca_features = self.pca.fit_transform(scaled_features)
        print("training model")
        self.kmeans = KMeans(n_clusters=CLUSTERS_NUM, random_state=0, n_init="auto")
        self.labels = self.kmeans.fit(pca_features)

    def infer(self, ratings: list[tuple[movieId, float]]) -> list[str]:
        array = [0 for _ in range(self.dataset.movies_count)]
        for movie, rating in ratings:
            array[self.dataset.movie_id_to_matrix_movie_id[movie]] = rating
        array = self.scaler.fit_transform([array])[0]
        array = self.pca.transform([array])[0]
        label = self.kmeans.predict([array])[0]
        print(label)
        print(self.get_movies_from_label(label))

    def get_movies_from_label(self, label: int):
        indices = np.where(self.labels.labels_ == label)[0]
        print(f"users : {len(indices)}")
        movies = self.matrix[indices]
        summed = np.sum(movies, axis=0)
        movies = []
        for i in range(self.dataset.movies_count):
            movies.append((i, summed[i]))
        movies = sorted(movies, key=lambda x: x[1], reverse=True)[0:10]
        matrix_ids = [self.dataset.matrix_movie_id_to_movie_id[a[0]] for a in movies]
        all_movies = self.dataset.get_all_movies()
        return [a for a in all_movies if a.movieId in matrix_ids], len(indices)


k = KMeansSystem()
# for fan in stereotypical_fans:
#     print(fan)
#     movies = [(movieId(a[0]), a[1]) for a in stereotypical_fans[fan]]
#     k.infer(movies)
counts = []
for i in range(CLUSTERS_NUM):
    print(f"label {i}")
    movies, count = k.get_movies_from_label(i)
    counts.append(count)
    c = Counter()
    for movie in movies:
        c.update(movie.genres)
    print(c.most_common())
    for movie in movies:
        print(f"{movie.title}, {movie.movieId}")
plt.plot(range(CLUSTERS_NUM), counts)
plt.xlabel("nr klastra")
plt.ylabel("ilość użytkowników")
plt.show()
