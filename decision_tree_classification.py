"""
ЧТО ТРЕБУЕТСЯ(кратко):
----------------------------------
1. На датасете load_wine() обучить дерево решений (DecisionTreeClassifier).
   При разбиении на train/test и в модели использовать random_state=17.
2. Отобразить дерево с помощью библиотеки graphviz.
3. С помощью GridSearchCV подобрать max_depth и max_features (5-fold CV),
   обучить tree_grid на тренировочных данных, вывести best_params_.
4. По полученной модели сделать predict для тестовой выборки и вывести
   точность (accuracy) через accuracy_score.

"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ══════════════════════════════════════════════════════════════════════════════

# Датасет Wine: 178 вин, 13 признаков (химический анализ), 3 класса.
from sklearn.datasets import load_wine

# DecisionTreeClassifier — алгоритм «дерево решений»: последовательность
# вопросов вида «признак X <= порог?», по ответам спускаемся к листу с классом.
from sklearn.tree import DecisionTreeClassifier

# train_test_split — разбивает данные на обучающую и тестовую части.
# Модель учится только на train, качество проверяем на test.
from sklearn.model_selection import train_test_split, GridSearchCV

# plot_tree — рисует дерево решений (работает через matplotlib, не требует Graphviz).
from sklearn.tree import plot_tree

# accuracy_score — считает долю правильных предсказаний (от 0 до 1).
from sklearn.metrics import accuracy_score

# matplotlib — для сохранения дерева в файл. Не требует установки Graphviz в PATH.
import matplotlib.pyplot as plt

# Константа для воспроизводимости: одно и то же «зерно» даёт одинаковую
# разбивку и случайность в модели при каждом запуске.
RANDOM_STATE = 17


def main():
    # ═══════════════════════════════════════════════════════════════════════
    # Загрузка данных, разбивка, обучение дерева решений
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  Загрузка данных и обучение дерева решений")
    print("=" * 60)

    # Загружаем датасет. data — таблица признаков, target — метки классов (0, 1, 2).
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Делим на обучающую и тестовую выборки. 75% — train, 25% — test.
    # random_state=17
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )

    # Создаём объект «дерево решений». Пока это просто алгоритм, он ещё не обучен.
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # Обучаем дерево на тренировочных данных. После fit() модель готова к predict.
    tree.fit(X_train, y_train)

    print("  Дерево обучено. Размер train:", X_train.shape[0], ", test:", X_test.shape[0])

    # ═══════════════════════════════════════════════════════════════════════
    # Отображение дерева (через matplotlib — без установки Graphviz)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Визуализация дерева")
    print("=" * 60)

    # Создаём фигуру большого размера, чтобы дерево поместилось и было читаемым.
    fig, ax = plt.subplots(figsize=(20, 12))
    # plot_tree рисует дерево на текущих осях.
    plot_tree(
        tree,
        feature_names=wine.feature_names,
        class_names=wine.target_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig("wine_tree.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  Дерево сохранено в файл wine_tree.png")

    # ═══════════════════════════════════════════════════════════════════════
    # GridSearchCV — подбор max_depth и max_features
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GridSearchCV, обучение, лучшие параметры")
    print("=" * 60)

    # Словарь параметров для перебора:
    # max_depth — максимальная глубина дерева (от 1 до 5).
    # max_features — сколько признаков рассматривать при каждом разбиении (от 1 до 9).
    tree_params = {
        "max_depth": range(1, 6),
        "max_features": range(1, 10),
    }

    # GridSearchCV перебирает все комбинации параметров, на каждой делает
    # 5-fold кросс-валидацию (cv=5) и запоминает лучшую комбинацию.
    # Первый аргумент — необученная модель (tree), второй — сетка параметров.
    tree_grid = GridSearchCV(tree, tree_params, cv=5)

    # Обучаем поиск по сетке на тренировочных данных.
    # Внутри он много раз обучает дерево с разными параметрами и выбирает лучшее.
    tree_grid.fit(X_train, y_train)

    # best_params_ — словарь с лучшей комбинацией параметров
    print("  Лучшие параметры (best_params_):", tree_grid.best_params_)

    # ═══════════════════════════════════════════════════════════════════════
    # Прогноз для тестовой выборки и точность (accuracy_score)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Прогноз на тесте и accuracy_score")
    print("=" * 60)

    # Полученная модель — это лучшая модель после GridSearchCV.
    # У GridSearchCV есть метод predict: он предсказывает с помощью лучшей модели
    # (внутри используется tree_grid.best_estimator_).
    predicted = tree_grid.predict(X_test)

    # accuracy_score сравнивает предсказания (predicted) с правильными
    # ответами (y_test) и возвращает долю совпадений (число от 0 до 1).
    accuracy = accuracy_score(y_test, predicted)

    # Выводим долю верных ответов
    print("  Доля верных ответов (accuracy):", accuracy)
    print("  (в процентах: {:.2f}%)".format(100 * accuracy))
    print("=" * 60)


if __name__ == "__main__":
    main()
