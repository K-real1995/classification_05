"""
Задача
------
Представлен алгоритм KNN (K ближайших соседей).
Теперь изучаем новый алгоритм классификации — наивный байесовский классификатор
(Gaussian Naive Bayes).

Что делает эта программа:
  1. Загружает датасет Wine (результаты химического анализа вин из Италии).
  2. Строит попарные графики признаков (scatter matrix), чтобы визуально
     выбрать лучшие 2 признака для классификации.
  3. Обучает модель GaussianNB на 2 лучших признаках и оценивает точность.
  4. Обучает модель GaussianNB на всех 13 признаках и сравнивает.
  5. Сравнивает GaussianNB с KNN на тех же данных.
  6. Визуализирует результаты классификации.

Что такое наивный байесовский классификатор?
--------------------------------------------
Это алгоритм, который использует теорему Байеса для предсказания класса.
Он называется «наивным», потому что считает все признаки НЕЗАВИСИМЫМИ
друг от друга (что в реальности редко бывает, но всё равно работает хорошо).

Гауссовский наивный Байес (GaussianNB) предполагает, что значения каждого
признака для каждого класса распределены по нормальному (гауссовскому)
закону — колоколообразная кривая. Для каждого класса алгоритм вычисляет:
  - среднее (μ) — «центр» распределения
  - дисперсию (σ²) — «ширину» распределения (разброс)

При предсказании алгоритм считает вероятность принадлежности объекта
к каждому классу и выбирает класс с наибольшей вероятностью.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# Импорт библиотек
# ══════════════════════════════════════════════════════════════

# numpy — для работы с числовыми массивами и математическими операциями
import numpy as np

# pandas — для работы с таблицами (DataFrame).
# DataFrame — это как Excel-таблица в Python: строки = объекты, столбцы = признаки.
import pandas as pd

# matplotlib — для построения графиков
import matplotlib.pyplot as plt

# load_wine — функция для загрузки знаменитого датасета Wine.
# Датасет содержит результаты химического анализа 178 вин из Италии,
# произведённых тремя разными виноделами (3 класса).
# У каждого вина 13 числовых признаков (содержание алкоголя, кислотность и т.д.).
from sklearn.datasets import load_wine

# train_test_split — разбивает данные на две части:
#   - тренировочная (на ней модель учится)
#   - тестовая (на ней проверяем качество модели)
# Это нужно, чтобы модель не «подсматривала» ответы при проверке.
from sklearn.model_selection import train_test_split

# GaussianNB — гауссовский наивный байесовский классификатор.
# Это основной алгоритм
from sklearn.naive_bayes import GaussianNB

# KNeighborsClassifier — алгоритм KNN (K ближайших соседей).
# Будем сравнивать с ним.
from sklearn.neighbors import KNeighborsClassifier

# accuracy_score — считает долю правильных предсказаний (от 0 до 1).
# Например, 0.95 значит, что модель угадала 95% объектов.
# classification_report — выводит подробный отчёт по каждому классу:
# precision, recall, f1-score (о них подробнее ниже).
from sklearn.metrics import accuracy_score, classification_report

# ══════════════════════════════════════════════════════════════
# Константы (настройки проекта)
# ══════════════════════════════════════════════════════════════

# random_state — «зерно» генератора случайных чисел.
# Фиксируем его, чтобы результаты были ВОСПРОИЗВОДИМЫМИ:
# каждый запуск программы даёт одинаковый результат.
# Значение 17 — стандартное.
RANDOM_STATE = 17

# Количество соседей для алгоритма KNN (для сравнения)
K_DEFAULT = 5

# Диапазон значений k для перебора (от 1 до 20 включительно)
K_RANGE = range(1, 21)

# ══════════════════════════════════════════════════════════════
# Два лучших признака для GaussianNB
# ══════════════════════════════════════════════════════════════
#
# Построили scatter_matrix (попарные графики) и визуально нашли
# признаки, которые ЛУЧШЕ ВСЕГО разделяют 3 класса вин.
#
# Для GaussianNB идеальные признаки — это те, у которых:
#   1) Значения для каждого класса распределены примерно симметрично
#      вокруг своего среднего (похожи на «колокольчик»).
#   2) «Облака» точек разных классов как можно МЕНЬШЕ пересекаются.
#
# После анализа scatter matrix выбираем:
#   - «flavanoids» (индекс 6) — содержание флавоноидов
#   - «od280/od315_of_diluted_wines» (индекс 11) — оптическая плотность
#
# Эти два признака хорошо разделяют все три класса вин,
# потому что у каждого класса значения этих признаков сильно различаются
# и мало пересекаются между собой.
BEST_FEATURE_1 = 6   # flavanoids
BEST_FEATURE_2 = 11  # od280/od315_of_diluted_wines


# ══════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════

def explore_dataset(wine_data) -> pd.DataFrame:
    """
    Исследует датасет Wine: выводит основную информацию и статистику.

    Зачем это нужно?
    Прежде чем обучать модель, нужно ПОНЯТЬ данные:
      - Сколько объектов? Сколько признаков?
      - Какие значения принимают признаки?
      - Есть ли пропуски в данных?

    Параметры
    ---------
    wine_data : объект датасета, загруженный через load_wine()

    Возвращает
    ----------
    df : DataFrame с данными (таблица pandas)
    """
    # Создаём DataFrame — удобную таблицу с названиями столбцов
    # wine_data.data — это числовая матрица (178 строк x 13 столбцов)
    # wine_data.feature_names — список названий 13 признаков
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

    print("=" * 70)
    print("  ИССЛЕДОВАНИЕ ДАТАСЕТА WINE")
    print("=" * 70)

    # Выводим названия всех 13 признаков
    print(f"\n  Признаки ({len(wine_data.feature_names)} шт.):")
    for i, name in enumerate(wine_data.feature_names):
        print(f"    [{i:2d}] {name}")

    # Выводим названия классов (3 сорта вина)
    print(f"\n  Классы: {wine_data.target_names}")

    # Выводим количество объектов каждого класса
    # np.unique считает, сколько раз встречается каждое уникальное значение
    unique_classes, counts = np.unique(wine_data.target, return_counts=True)
    print(f"  Распределение классов:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"    Класс {cls} ({wine_data.target_names[cls]}): {cnt} объектов")

    # Выводим общую информацию о таблице
    print(f"\n  Размер таблицы: {df.shape[0]} строк x {df.shape[1]} столбцов")

    # Выводим статистику по каждому признаку:
    # count (кол-во), mean (среднее), std (стд. отклонение), min, max и т.д.
    print("\n  Статистика по признакам:")
    print(df.describe().to_string())

    return df


def plot_scatter_matrix(df: pd.DataFrame, target: np.ndarray) -> None:
    """
    Строит матрицу попарных графиков (scatter matrix).

    Что это такое?
    Это сетка графиков, где каждый маленький график показывает
    зависимость между двумя признаками. На диагонали — гистограммы
    распределения каждого признака.

    Зачем это нужно?
    Помогает ВИЗУАЛЬНО найти признаки, которые лучше всего
    разделяют классы. Если «облака» точек разных цветов
    (= разных классов) хорошо отделены друг от друга на графике,
    значит эти признаки — хорошие кандидаты для классификации.

    Параметры
    ---------
    df     : DataFrame с признаками
    target : массив меток классов (0, 1 или 2)
    """
    print("\n  Строим scatter matrix (попарные графики)...")
    print("  Это может занять несколько секунд — строится 13x13 = 169 графиков!")

    # Задаём размер фигуры 25x25 дюймов (как рекомендовано в задании)
    # c=target — цвет точки определяется классом вина
    # alpha=0.5 — полупрозрачные точки (чтобы видеть наложения)
    # figsize=(25, 25) — большой размер для удобства изучения
    pd.plotting.scatter_matrix(
        df,
        c=target,
        figsize=(25, 25),
        alpha=0.5,
        diagonal="hist",  # на диагонали рисуем гистограммы
        s=30,             # размер точек
    )

    plt.suptitle(
        "Scatter Matrix — попарные графики признаков Wine",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )


def plot_selected_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    idx1: int,
    idx2: int,
) -> None:
    """
    Строит крупный 2D-график для двух выбранных признаков.

    Каждая точка — одно вино. Цвет точки — класс вина.
    Этот график показывает, ПОЧЕМУ мы выбрали именно эти два признака:
    три «облака» точек хорошо разделены.

    Параметры
    ---------
    X             : матрица всех признаков
    y             : массив меток классов
    feature_names : список названий признаков
    idx1, idx2    : индексы двух выбранных признаков
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Рисуем точки каждого класса отдельно, чтобы добавить подписи в легенду
    # Названия классов из датасета: class_0, class_1, class_2
    wine = load_wine()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # синий, оранжевый, зелёный

    for class_idx in range(3):
        # Выбираем только строки, которые принадлежат текущему классу
        mask = y == class_idx
        ax.scatter(
            X[mask, idx1],
            X[mask, idx2],
            c=colors[class_idx],
            label=wine.target_names[class_idx],
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
            s=60,
        )

    ax.set_xlabel(feature_names[idx1], fontsize=12)
    ax.set_ylabel(feature_names[idx2], fontsize=12)
    ax.set_title(
        f"Два лучших признака для GaussianNB:\n"
        f"«{feature_names[idx1]}» и «{feature_names[idx2]}»",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, title="Класс вина")
    ax.grid(True, alpha=0.3)


def train_gaussian_nb(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[GaussianNB, float]:
    """
    Обучает модель Gaussian Naive Bayes и возвращает (модель, точность).

    Как работает GaussianNB?
    1) Для каждого класса и каждого признака алгоритм считает:
       - среднее значение (μ)
       - стандартное отклонение (σ)
    2) При предсказании для нового объекта алгоритм вычисляет
       вероятность того, что этот объект принадлежит каждому классу,
       используя формулу нормального распределения.
    3) Выбирается класс с НАИБОЛЬШЕЙ вероятностью.

    Преимущества GaussianNB:
      - Очень быстрый (обучение за миллисекунды)
      - Хорошо работает на небольших датасетах
      - Не нужно подбирать гиперпараметры (в отличие от KNN)

    Параметры
    ---------
    X_train : признаки для обучения
    X_test  : признаки для проверки
    y_train : правильные ответы для обучения
    y_test  : правильные ответы для проверки

    Возвращает
    ----------
    model    : обученная модель GaussianNB
    accuracy : точность на тестовой выборке (от 0.0 до 1.0)
    """
    # Создаём модель GaussianNB
    # У неё НЕТ гиперпараметров, которые нужно настраивать вручную!
    # Это одно из преимуществ перед KNN, где нужно подбирать k.
    model = GaussianNB()

    # Обучаем модель на тренировочных данных.
    # Внутри fit() алгоритм считает среднее и дисперсию
    # для каждого признака в каждом классе.
    model.fit(X_train, y_train)

    # Делаем предсказания на тестовых данных и считаем точность
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def train_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = K_DEFAULT,
) -> tuple[KNeighborsClassifier, float]:
    """
    Обучает модель KNN и возвращает (модель, точность).

    Напоминание: KNN запоминает все тренировочные данные.
    Для нового объекта ищет k ближайших соседей
    и выбирает самый частый класс среди них.

    Параметры
    ---------
    X_train, X_test, y_train, y_test : данные для обучения/проверки
    n_neighbors : количество соседей (k)

    Возвращает
    ----------
    model    : обученная модель KNN
    accuracy : точность на тестовой выборке
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


def find_best_k_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_range: range = K_RANGE,
) -> tuple[list[int], float, list[float]]:
    """
    Перебирает значения k от 1 до 20 и находит лучшее для KNN.

    Зачем?
    В KNN результат сильно зависит от выбора k:
      - k=1: модель «переобучается» — слишком чувствительна к шуму
      - k=20: модель слишком «грубая» — может упустить закономерности
    Нужно найти «золотую середину».

    Возвращает
    ----------
    best_ks    : список лучших значений k
    best_acc   : лучшая точность
    accuracies : список точностей для каждого k (для графика)
    """
    best_acc = -1.0
    best_ks: list[int] = []
    accuracies: list[float] = []

    for k in k_range:
        _, acc = train_knn(X_train, X_test, y_train, y_test, n_neighbors=k)
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_ks = [k]
        elif acc == best_acc:
            best_ks.append(k)

    return best_ks, best_acc, accuracies


def plot_comparison(results: dict[str, float]) -> None:
    """
    Строит столбчатую диаграмму для сравнения точности разных моделей.

    Это итоговый график, который наглядно показывает,
    какая модель (и с какими признаками) работает лучше.

    Параметры
    ---------
    results : словарь {название_модели: точность}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(results.keys())
    accs = list(results.values())

    # Рисуем горизонтальные столбцы
    bars = ax.barh(names, accs, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    # Подписываем значение точности рядом с каждым столбцом
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Точность (Accuracy)", fontsize=12)
    ax.set_title(
        "Сравнение моделей классификации на датасете Wine",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis="x", alpha=0.3)


def plot_k_accuracy(k_range: range, accuracies: list[float], title: str) -> None:
    """
    Строит график зависимости точности KNN от количества соседей k.

    По оси X — значение k (от 1 до 20).
    По оси Y — точность модели.
    Помогает визуально найти оптимальное значение k.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(list(k_range), accuracies, marker="o", linewidth=2, markersize=6)

    ax.set_xlabel("Количество соседей (k)", fontsize=12)
    ax.set_ylabel("Точность (Accuracy)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3)

    # Ограничиваем ось Y, чтобы различия были лучше видны
    min_acc = min(accuracies)
    ax.set_ylim(max(0, min_acc - 0.05), 1.02)


def plot_decision_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: tuple[str, str],
) -> None:
    """
    Строит 2 графика рядом: предсказания GaussianNB и KNN на 2 признаках.

    Каждая точка — одно вино из тестовой выборки.
    Правильные предсказания — кружки, неправильные — крестики.
    Это помогает ВИЗУАЛЬНО увидеть, где модели ошибаются.

    Параметры
    ---------
    X_train, X_test  : признаки для обучения и проверки
    y_train, y_test  : метки классов
    feature_names    : названия двух признаков (для подписей осей)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Обучаем обе модели
    gnb = GaussianNB().fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=K_DEFAULT).fit(X_train, y_train)

    models = [("GaussianNB", gnb), (f"KNN (k={K_DEFAULT})", knn)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (model_name, model) in zip(axes, models):
        y_pred = model.predict(X_test)

        for class_idx in range(3):
            # Находим правильные предсказания для текущего класса
            correct = (y_test == class_idx) & (y_pred == class_idx)
            # Находим неправильные предсказания
            wrong = (y_test == class_idx) & (y_pred != class_idx)

            # Рисуем правильные предсказания кружками
            ax.scatter(
                X_test[correct, 0],
                X_test[correct, 1],
                c=colors[class_idx],
                marker="o",
                s=80,
                alpha=0.8,
                edgecolors="k",
                linewidths=0.5,
                label=f"Класс {class_idx} (верно)",
            )

            # Рисуем неправильные предсказания крестиками (если есть)
            if np.sum(wrong) > 0:
                ax.scatter(
                    X_test[wrong, 0],
                    X_test[wrong, 1],
                    c=colors[class_idx],
                    marker="x",
                    s=100,
                    linewidths=2,
                    label=f"Класс {class_idx} (ошибка)",
                )

        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{model_name}\nТочность: {acc:.4f}", fontsize=13, fontweight="bold")
        ax.set_xlabel(feature_names[0], fontsize=11)
        ax.set_ylabel(feature_names[1], fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Сравнение предсказаний GaussianNB и KNN (тестовая выборка)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()


# ══════════════════════════════════════════════════════════════
# Главная функция (основной конвейер программы)
# ══════════════════════════════════════════════════════════════

def main() -> None:
    # =========================================================
    # ШАГ 1. Загружаем и изучаем датасет Wine
    # =========================================================
    # Датасет Wine содержит 178 образцов вин из Италии.
    # Три класса — три разных производителя (культиватора).
    # У каждого вина 13 числовых признаков:
    #   [0] alcohol                       — содержание алкоголя (%)
    #   [1] malic_acid                    — яблочная кислота
    #   [2] ash                           — зольность
    #   [3] alcalinity_of_ash             — щёлочность золы
    #   [4] magnesium                     — магний
    #   [5] total_phenols                 — общее количество фенолов
    #   [6] flavanoids                    — флавоноиды
    #   [7] nonflavanoid_phenols          — нефлавоноидные фенолы
    #   [8] proanthocyanins               — проантоцианины
    #   [9] color_intensity               — интенсивность цвета
    #   [10] hue                          — оттенок
    #   [11] od280/od315_of_diluted_wines — оптическая плотность
    #   [12] proline                      — пролин (аминокислота)
    wine = load_wine()

    # X_full — таблица всех признаков (178 строк x 13 столбцов)
    # y      — массив меток классов (178 значений: 0, 1 или 2)
    X_full = wine.data
    y = wine.target

    # Изучаем датасет — смотрим на размеры, статистику, распределение классов
    df = explore_dataset(wine)

    # =========================================================
    # ШАГ 2. Строим scatter matrix — попарные графики
    # =========================================================
    # Это ключевой шаг для ЗАДАНИЯ 1.
    # На scatter matrix мы визуально ищем пары признаков,
    # которые лучше всего разделяют три класса.
    #
    # Для GaussianNB лучшие признаки — те, у которых:
    #   - Распределение похоже на нормальное (колоколообразное)
    #   - «Облака» разных классов НЕ пересекаются
    #
    # После анализа мы выбрали: flavanoids и od280/od315_of_diluted_wines
    plot_scatter_matrix(df, y)

    # Строим отдельный крупный график для двух выбранных признаков
    plot_selected_features(
        X_full, y, wine.feature_names,
        BEST_FEATURE_1, BEST_FEATURE_2,
    )

    # =========================================================
    # ШАГ 3. Готовим наборы данных
    # =========================================================
    # Набор 1: только два лучших признака (flavanoids + od280/od315)
    X_2features = X_full[:, [BEST_FEATURE_1, BEST_FEATURE_2]]
    selected_names = (
        wine.feature_names[BEST_FEATURE_1],
        wine.feature_names[BEST_FEATURE_2],
    )

    # Набор 2: все 13 признаков
    X_all = X_full

    # =========================================================
    # ШАГ 4. Разбиваем данные на тренировочную и тестовую выборки
    # =========================================================
    # train_test_split делит данные случайным образом:
    #   - 75% данных — для обучения (train)
    #   - 25% данных — для проверки (test)
    # random_state=17 фиксирует случайность для воспроизводимости.

    # Разбивка для 2 признаков
    X_train_2f, X_test_2f, y_train_2f, y_test_2f = train_test_split(
        X_2features, y, random_state=RANDOM_STATE,
    )

    # Разбивка для всех 13 признаков
    # ВАЖНО: используем ОДИНАКОВЫЙ random_state, чтобы разбивка была идентичной!
    # Это позволяет честно сравнивать модели — они обучаются и тестируются
    # на ОДНИХ И ТЕХ ЖЕ объектах.
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y, random_state=RANDOM_STATE,
    )

    # =========================================================
    # ШАГ 5. Обучаем GaussianNB на 2 признаках
    # =========================================================
    print("\n" + "=" * 70)
    print("  РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 70)

    gnb_2f_model, gnb_2f_acc = train_gaussian_nb(
        X_train_2f, X_test_2f, y_train_2f, y_test_2f,
    )

    print(f"\n>> GaussianNB (2 признака: {selected_names[0]}, {selected_names[1]})")
    print(f"   Точность: {gnb_2f_acc:.4f}")
    print(f"   Подробный отчёт:")
    print(classification_report(
        y_test_2f,
        gnb_2f_model.predict(X_test_2f),
        target_names=wine.target_names,
    ))

    # =========================================================
    # ШАГ 6. Обучаем GaussianNB на всех 13 признаках
    # =========================================================
    gnb_all_model, gnb_all_acc = train_gaussian_nb(
        X_train_all, X_test_all, y_train_all, y_test_all,
    )

    print(f">> GaussianNB (все 13 признаков)")
    print(f"   Точность: {gnb_all_acc:.4f}")
    print(f"   Подробный отчёт:")
    print(classification_report(
        y_test_all,
        gnb_all_model.predict(X_test_all),
        target_names=wine.target_names,
    ))

    # =========================================================
    # ШАГ 7. Сравниваем с KNN (на тех же данных)
    # =========================================================
    # Обучаем KNN (k=5) на 2 признаках
    knn_2f_model, knn_2f_acc = train_knn(
        X_train_2f, X_test_2f, y_train_2f, y_test_2f,
    )

    print(f">> KNN, k={K_DEFAULT} (2 признака: {selected_names[0]}, {selected_names[1]})")
    print(f"   Точность: {knn_2f_acc:.4f}")
    print(f"   Подробный отчёт:")
    print(classification_report(
        y_test_2f,
        knn_2f_model.predict(X_test_2f),
        target_names=wine.target_names,
    ))

    # Обучаем KNN (k=5) на всех 13 признаках
    knn_all_model, knn_all_acc = train_knn(
        X_train_all, X_test_all, y_train_all, y_test_all,
    )

    print(f">> KNN, k={K_DEFAULT} (все 13 признаков)")
    print(f"   Точность: {knn_all_acc:.4f}")
    print(f"   Подробный отчёт:")
    print(classification_report(
        y_test_all,
        knn_all_model.predict(X_test_all),
        target_names=wine.target_names,
    ))

    # =========================================================
    # ШАГ 8. Ищем лучшее k для KNN (для полноты сравнения)
    # =========================================================
    # Перебираем k от 1 до 20 для KNN на 2 признаках
    best_ks_2f, best_acc_2f, accs_2f = find_best_k_knn(
        X_train_2f, X_test_2f, y_train_2f, y_test_2f,
    )

    print("-" * 70)
    print(f"  Лучшие k для KNN (2 признака): {best_ks_2f}  (точность = {best_acc_2f:.4f})")

    # Перебираем k от 1 до 20 для KNN на всех признаках
    best_ks_all, best_acc_all, accs_all = find_best_k_knn(
        X_train_all, X_test_all, y_train_all, y_test_all,
    )

    print(f"  Лучшие k для KNN (все признаки): {best_ks_all}  (точность = {best_acc_all:.4f})")
    print("-" * 70)

    # =========================================================
    # ШАГ 9. Итоговое сравнение всех моделей
    # =========================================================
    print("\n" + "=" * 70)
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 70)

    # Собираем все результаты в словарь
    results = {
        f"GaussianNB (2 признака)": gnb_2f_acc,
        f"GaussianNB (13 признаков)": gnb_all_acc,
        f"KNN, k={K_DEFAULT} (2 признака)": knn_2f_acc,
        f"KNN, k={K_DEFAULT} (13 признаков)": knn_all_acc,
    }

    # Выводим результаты в виде таблицы
    for name, acc in results.items():
        print(f"   {name:40s} -> {acc:.4f}")

    # Находим лучшую модель
    best_model_name = max(results, key=results.get)
    print(f"\n   Лучшая модель: {best_model_name} ({results[best_model_name]:.4f})")

    # =========================================================
    # ШАГ 10. Визуализация результатов
    # =========================================================

    # 10.1 Столбчатая диаграмма — сравнение точности всех моделей
    plot_comparison(results)

    # 10.2 Графики зависимости точности KNN от k
    plot_k_accuracy(
        K_RANGE, accs_2f,
        f"Зависимость точности KNN от k (2 признака)",
    )
    plot_k_accuracy(
        K_RANGE, accs_all,
        f"Зависимость точности KNN от k (все 13 признаков)",
    )

    # 10.3 Сравнение предсказаний GaussianNB и KNN на тестовой выборке
    plot_decision_comparison(
        X_train_2f, X_test_2f, y_train_2f, y_test_2f,
        feature_names=selected_names,
    )

    # =========================================================
    # ШАГ 11. Выводы
    # =========================================================
    print("\n" + "=" * 70)
    print("  ВЫВОДЫ")
    print("=" * 70)
    print("""
    1. GaussianNB — простой и быстрый алгоритм. Он не требует подбора
       гиперпараметров (в отличие от KNN, где нужно выбирать k).

    2. Даже на 2 правильно выбранных признаках GaussianNB может давать
       хорошую точность. Правильный ВЫБОР ПРИЗНАКОВ часто важнее,
       чем усложнение модели.

    3. Использование всех 13 признаков может как улучшить, так и ухудшить
       результат — это зависит от данных. Лишние «шумные» признаки
       могут НАВРЕДИТЬ модели (особенно GaussianNB, который предполагает
       независимость признаков).

    4. KNN и GaussianNB — два разных подхода к классификации:
       - KNN запоминает все данные и ищет ближайших соседей
       - GaussianNB строит статистическую модель (среднее + дисперсия)
       Какой лучше — зависит от конкретной задачи и данных.
    """)

    # Подгоняем отступы, чтобы графики не наезжали друг на друга
    plt.tight_layout()

    # Показываем все графики
    plt.show()


# ══════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════
# Эта конструкция означает: «Запускай main() только если файл
# запущен напрямую (python main.py), а не импортирован как модуль
# в другой файл».
# Это стандартная практика в Python-проектах.
if __name__ == "__main__":
    main()
