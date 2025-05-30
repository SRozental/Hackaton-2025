# Задача: Прогнозирование оттока клиентов
## Партнер: Ростелеком
### Команда: Анонимус2008

## Описание проекта
Проект посвящен решению задачи прогнозирования оттока клиентов. На основе исторических данных о поведении клиентов мы выявили ключевые тренды и построили модель машинного обучения (Random Forest), которая с высокой точностью предсказывает вероятность совершения следующей покупки.

```bash
## Структура репозитория
├── Data/ # Исходные и обработанные данные
|
├── notebooks/ # Jupyter Notebooks с анализом и моделированием
│ ├── EDA.ipynb # Разведочный анализ данных
│ ├── feature_engineering.ipynb # Генерация признаков, обучение модели Random Forest
| ├── Customers distribution maps.ipynb
| ├── geography.ipynb # Анализ географии пользователей
│ └── reviews.py # Анализ рейтингов
| └── Customers distribution maps.ipynb #Построение карт распределения пользователей
|
├── results/ # результаты
│ ├── presentation.pdf # Презентация
└── README.md # Этот файл
```


## Ключевые результаты
1. **Выявленные тренды**:
   - Только 3% клиентов совершают вторую покупку, а третью еще в 10 раз меньше
   - Стоимость доставки составляет в среднем 30-35% от стоимости товара, что может являться одним из ключевых фактороы оттока клиентов.

2. **Модель Random Forest**:
   - F1: 99.8%
   - Построена Confusion matrix

3. **Важные признаки**:
   - Время с последней покупки
   - Время доставки
   - Рейтинг покупки
     
4. **Дашборд**:
    https://datalens.yandex.cloud/hs5v7mc7njae3

Участники команды
- Петр (Ведущий аналитик)
- Лидия (BI-аналитик)
- Алексей (ML-разработчик)
- Сусанна (Аналитик данных)
- Иван (Аналитик данных)
- Владимир (Аналитик данных)
