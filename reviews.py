import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Конфигурация путей и файлов
data_dir = 'data'
required_files = {
    'reviews': 'order_reviews.csv',
    'orders': 'orders.csv',
    'items': 'orders_items.csv',
    'products': 'products.csv',
    'translation': 'product_category_name_translation.csv'
}

# Проверка наличия папки и файлов
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Папка '{data_dir}' создана. Поместите в нее следующие файлы:")
    for f in required_files.values():
        print(f"- {f}")
    exit()

missing_files = [f for f in required_files.values() if not os.path.exists(os.path.join(data_dir, f))]
if missing_files:
    print(f"В папке '{data_dir}' отсутствуют файлы:")
    for f in missing_files:
        print(f"- {f}")
    print("\nПроверьте:")
    print("1. Правильность названий файлов")
    print("2. Что файлы имеют расширение .csv")
    print("3. Что файлы не открыты в других программах")
    exit()

print("Все файлы найдены, начинаем обработку...")

try:
    # Загрузка данных с указанием типов для проблемных столбцов
    reviews = pd.read_csv(os.path.join(data_dir, required_files['reviews']),
                          usecols=['review_id', 'order_id', 'review_score'])

    orders = pd.read_csv(os.path.join(data_dir, required_files['orders']),
                         usecols=['order_id'])

    items = pd.read_csv(os.path.join(data_dir, required_files['items']),
                        usecols=['order_id', 'product_id'],
                        dtype={'order_id': str, 'product_id': str})

    products = pd.read_csv(os.path.join(data_dir, required_files['products']),
                           usecols=['product_id', 'product_category_name'],
                           dtype={'product_id': str})

    # Загрузка таблицы перевода категорий
    category_translation = pd.read_csv(os.path.join(data_dir, required_files['translation']),
                                       usecols=['product_category_name', 'product_category_name_english'])

    # Объединение данных по цепочке
    # Шаг 1: Отзывы + заказы
    merged = pd.merge(reviews, orders, on='order_id')

    # Шаг 2: Добавляем информацию о товарах
    merged = pd.merge(merged, items, on='order_id')

    # Шаг 3: Добавляем категории товаров
    merged = pd.merge(merged, products, on='product_id')

    # Шаг 4: Добавляем перевод категорий
    merged = pd.merge(merged, category_translation, on='product_category_name', how='left')

    # Заменяем NaN значения на оригинальные названия категорий
    merged['product_category_name_english'] = merged['product_category_name_english'].fillna(
        merged['product_category_name'])

    # Анализ данных
    category_stats = merged.groupby('product_category_name_english')['review_score'] \
        .agg(['mean', 'count']) \
        .rename(columns={'mean': 'avg_rating', 'count': 'reviews_count'}) \
        .sort_values('avg_rating', ascending=False)

    # Фильтр для значимых категорий (более 10 отзывов)
    significant_categories = category_stats[category_stats['reviews_count'] > 10]

    # Визуализация
    plt.figure(figsize=(14, 10))
    sns.barplot(data=significant_categories.reset_index(),
                x='avg_rating',
                y='product_category_name_english',
                hue='product_category_name_english',
                palette='coolwarm',
                legend=False)

    plt.title('Average Product Ratings by Category\n(Categories with 10+ reviews only)', pad=20)
    plt.xlabel('Average Rating (1-5)', labelpad=15)
    plt.ylabel('Product Category', labelpad=15)
    plt.xlim(0, 5)
    plt.grid(axis='x', alpha=0.3)

    # Сохранение результатов
    os.makedirs('results', exist_ok=True)

    # Сохраняем график
    plt.savefig('results/category_ratings_en.png', dpi=300, bbox_inches='tight')

    # Сохраняем таблицу с рейтингами
    significant_categories.to_csv('results/category_ratings_en.csv')

    print("\nAnalysis completed successfully!")
    print(f"Saved results:")
    print(f"- Chart: results/category_ratings_en.png")
    print(f"- Data: results/category_ratings_en.csv")
    plt.show()

except Exception as e:
    print(f"\nError processing data: {str(e)}")
    print("\nPossible reasons:")
    print("- Column name mismatch in files")
    print("- Data format issues in CSV files")
    print("- Corrupted files")
    print("\nTry the following:")
    print("1. Check CSV files in a text editor")
    print("2. Verify all files use the same delimiter (usually comma)")
    print("3. Check for problematic characters in data")