import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Настройка страницы
st.set_page_config(
    page_title="Анализ оттока клиентов",
    page_icon="📊",
    layout="wide"
)

# Словарь для русскоязычных названий
russian_labels = {
    # Колонки данных
    'Churn': 'Отток клиентов',
    'tenure': 'Срок обслуживания (мес)',
    'MonthlyCharges': 'Ежемесячные платежи',
    'TotalCharges': 'Общие платежи',
    'customerID': 'ID клиента',

    # Значения
    'Yes': 'Да',
    'No': 'Нет',

    # Интерфейс
    'Data': 'Данные',
    'Analysis': 'Анализ',
    'Model': 'Модель',
    'Settings': 'Настройки',
    'Metrics': 'Метрики',
    'Feature Importance': 'Важность признаков'
}

# Заголовок
st.title("📊 Анализ оттока клиентов телеком-компании")
st.markdown("Анализ данных и прогнозирование оттока клиентов на основе исторических данных")


# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\Iliya\PycharmProjects\ilyailya\НИР\data.csv')

        # Проверка обязательных колонок
        required_cols = ['Churn', 'tenure', 'MonthlyCharges']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Отсутствует обязательная колонка: {col}")
                return None

        # Обработка данных
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df = df.dropna(subset=['TotalCharges'])

        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None


# Загружаем данные
df = load_data()
if df is None:
    st.stop()

# Боковая панель
st.sidebar.header("⚙️ Настройки модели")
test_size = st.sidebar.slider("Размер тестовой выборки (%)", 10, 40, 20)
n_estimators = st.sidebar.slider("Количество деревьев", 50, 200, 100)
max_depth = st.sidebar.slider("Глубина деревьев", 2, 20, 5)

# Вкладки
tab1, tab2, tab3 = st.tabs(["📁 Данные", "📊 Анализ", "🤖 Модель"])

with tab1:
    st.header("1. Исходные данные")
    st.write(f"Всего записей: {len(df)}")

    # Отображение данных
    st.dataframe(df.head(10).rename(columns=russian_labels), height=300)

    # Статистика
    st.subheader("Основные статистические показатели")
    st.dataframe(df.describe().rename(columns=russian_labels))

with tab2:
    st.header("2. Анализ данных")

    # Распределение оттока
    st.subheader("Распределение оттока клиентов")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    plot = sns.countplot(data=df, x='Churn', palette='viridis', ax=ax1)
    plot.set(xlabel=russian_labels['Churn'],
             ylabel='Количество клиентов',
             title='Распределение клиентов по оттоку')
    plot.set_xticklabels([russian_labels.get(x.get_text(), x.get_text()) for x in plot.get_xticklabels()])
    st.pyplot(fig1)

    # Распределение числовых признаков
    st.subheader("Распределение числовых показателей")
    num_col = st.selectbox("Выберите показатель",
                           [('tenure', 'Срок обслуживания'),
                            ('MonthlyCharges', 'Ежемесячные платежи'),
                            ('TotalCharges', 'Общие платежи')],
                           format_func=lambda x: x[1])[0]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot = sns.histplot(data=df, x=num_col, hue='Churn', kde=True, palette='viridis', ax=ax2)
    plot.set(xlabel=russian_labels.get(num_col, num_col),
             ylabel='Количество клиентов',
             title=f'Распределение {russian_labels.get(num_col, num_col)} по оттоку')
    plt.legend(title=russian_labels['Churn'], labels=[russian_labels['No'], russian_labels['Yes']])
    st.pyplot(fig2)

with tab3:
    st.header("3. Прогнозирование оттока")


    # Предобработка данных
    def preprocess_data(df):
        df = df.copy()

        # Удаление лишних колонок
        df = df.drop(['customerID', 'Unnamed: 0'], axis=1, errors='ignore')

        # Кодирование целевой переменной
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})

        # Кодирование категориальных признаков
        cat_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        # Масштабирование числовых признаков
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['Churn'])
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        return df


    processed_df = preprocess_data(df)

    # Разделение данных
    X = processed_df.drop('Churn', axis=1)
    y = processed_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y
    )

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Метрики
    st.subheader("Оценка качества модели")
    col1, col2, col3 = st.columns(3)
    col1.metric("Точность (Accuracy)", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("F1-мера", f"{f1_score(y_test, y_pred):.3f}")
    col3.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

    # Матрица ошибок
    st.subheader("Матрица ошибок классификации")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Остался', 'Ушел'],
                yticklabels=['Остался', 'Ушел'])
    ax3.set_xlabel("Предсказанный класс")
    ax3.set_ylabel("Истинный класс")
    st.pyplot(fig3)

    # Важность признаков
    st.subheader("Важность признаков для модели")
    importances = pd.DataFrame({
        'Признак': X.columns,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False).head(10)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importances, x='Важность', y='Признак', ax=ax4, palette='viridis')
    ax4.set_title("Топ-10 наиболее важных признаков")
    ax4.set_xlabel("Важность")
    ax4.set_ylabel("Признак")
    st.pyplot(fig4)

# Подвал
st.sidebar.markdown("---")
st.sidebar.info("""
**Информация:**  
Приложение для анализа оттока клиентов.  
Данные загружаются из локального файла.
""")
