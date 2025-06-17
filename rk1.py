import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = pd.read_csv('toy_dataset.csv')

# Добавляем 20% пропусков в столбец 'Age'
mask_age = np.random.choice([True, False], size=len(data), p=[0.2, 0.8])
data.loc[mask_age, 'Age'] = np.nan

# Добавляем 15% пропусков в столбец 'Income'
mask_income = np.random.choice([True, False], size=len(data), p=[0.15, 0.85])
data.loc[mask_income, 'Income'] = np.nan

print("Данные с добавленными пропусками:")
print(data)

print("\nКоличество пропусков по столбцам:")
print(data.isnull().sum())

data_cleaned = data.dropna()
print("\nДанные после удаления пропусков:")
print(data_cleaned)

# Заполняем пропуски в 'Age' средним возрастом
data_filled = data.copy()
data_filled['Age'] = data_filled['Age'].fillna(data['Age'].mean())
data_filled['Income'] = data_filled['Income'].fillna(data['Income'].median())

print("\nКоличество пропусков по столбцам:")
print(data_filled.isnull().sum())

print("\nДанные после заполнения пропусков:")
print(data_filled)

# Выбор числовых столбцов Расчет корреляции Визуализация
numeric_data = data_filled.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()
data.drop('Age', axis=1, inplace=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Корреляционная матрица")
plt.show()

# Выбор колонок
col1 = 'Age'
col2 = 'Income'

# Построение Jointplot
sns.jointplot(
    x=col1,
    y=col2,
    data=data_filled,
    kind='scatter',
    height=7,
    alpha=0.6  # Прозрачность точек
)
plt.suptitle(f'Зависимость между {col1} и {col2}', y=1.02)
plt.tight_layout()
plt.savefig('jointplot.png')  # Сохранение графика
plt.show()
