import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset

d = Dataset()
meta = d.get_movie_meta()  # rating, count
counts = sorted([meta[m][1] for m in meta], reverse=True)
print(sum(counts[:100]))
print(sum(counts[100:]))
sns.set_style('whitegrid')
k = sns.lineplot(counts)
k.set(xlabel='Film', ylabel='Ilość ocen')
plt.show()
