import random
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import Dataset

import seaborn as sns


def accuracy():
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 100)
    jaccard_indexes = []
    for user in tqdm(test_users):
        user_label = d.label_by_user[user]
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:10]
        if len(top_movies) == 10:
            liked_movies_set = {a[0] for a in top_movies}
            users_in_group = [u for u in d.label_by_user if d.label_by_user[u] == user_label]
            ratings_in_group = []
            for user_from_group in users_in_group:
                ratings_in_group += d.ratings_by_user[user_from_group]
            counter = Counter()
            for m_id, rating in ratings_in_group:
                counter[m_id] += rating
            recommendations_set = {a[0] for a in counter.most_common()[0:10]}
            jaccard_index = len(liked_movies_set.intersection(recommendations_set)) / len(liked_movies_set.union(recommendations_set))
            jaccard_indexes.append(jaccard_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Indeks Jaccarda', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.show()


def variety():
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    all_vectors = [movie.genres_vector for movie in d.all_movies]
    global_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*all_vectors)]
    global_average = [i / sum(global_average) for i in global_average]
    for user in tqdm(test_users):
        user_label = d.label_by_user[user]
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:10]
        if len(top_movies) == 10:
            users_in_group = [u for u in d.label_by_user if d.label_by_user[u] == user_label]
            ratings_in_group = []
            for user_from_group in users_in_group:
                ratings_in_group += d.ratings_by_user[user_from_group]
            counter = Counter()
            for m_id, rating in ratings_in_group:
                counter[m_id] += rating
            recommendations_set = {a[0] for a in counter.most_common()[0:10]}
            recommendations = [movie.genres_vector for movie in d.all_movies if movie.movieId in recommendations_set]
            column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*recommendations)]
            variety_index = sum([(column_average[i] - global_average[i]) ** 2 for i in range(len(column_average))])
            jaccard_indexes.append(variety_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Różnorodność', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.show()


variety()
