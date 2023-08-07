import random
from collections import defaultdict

from dataset import Dataset, User, userId
from math import dist


def get_distance(u1: User, u2: User) -> float:
    return dist(u1.vector, u2.vector)


def movies_from_users(d: Dataset, users: list[userId], watched):
    ratings = d.ratings_by_user
    ddict = defaultdict(lambda: 0.0)
    movies_map = {a.movieId: a for a in d.all_movies}
    for user in users:
        for m_id, rating in ratings[user]:
            ddict[m_id] += rating
    top = sorted(ddict.items(), key=lambda item: item[1], reverse=True)
    return [movies_map[a[0]].title for a in top if movies_map[a[0]].movieId not in watched][0:10]


d = Dataset()
all_users = d.get_all_user_ids()
test_users = random.sample(all_users, len(all_users) // 100)
t_user_id = test_users[0]
print(t_user_id)
t_user_label = d.label_by_user[t_user_id]
users = d.get_user_vectors(t_user_label)
for user in users:
    print(user.id)
    print(sum(user.vector))
t_user = [u for u in users if u.id == t_user_id][0]
print(len(users))
ranked_users = [User(u.id, u.vector, get_distance(t_user, u)) for u in users]
sorted_list = sorted(ranked_users, key=lambda x: x.rank, reverse=True)
top = [a.id for a in sorted_list if a.id != t_user][:3]
print(top)
watched = d.ratings_by_user[t_user.id]
print(movies_from_users(d, top, watched))
