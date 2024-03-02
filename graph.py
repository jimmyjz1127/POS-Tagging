import pandas as pd
# from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 




df = pd.read_csv('./confusion_matrix_data.csv', index_col=0)

sum_row = df.sum(axis=1)
df = df.div(sum_row, axis=0)

plt.figure(figsize=(10, 8))
sns.heatmap(df, fmt='.2f', cmap='Reds', annot=True)
plt.title('Confusion')
plt.show()

