### Salary_predictor

Довольно часто работодатель не указывает з/п в вакансии. Этот проект позволяет решить данную проблему, предсказывая ее по описанию вакансии.
### Используемый стек:
    - Python (transformers, Pytorch, pandas, FastAPI)
    - Docker & Docker compose
    - Git & GitHub
    - Amvera Cloud для развертывания
### Используемые сторонние ресурсы:
    - RuBERT(tokenizer & BERT), Kaggle dataset, Kaggle notebook (для обучения модели)

### Навигация по проекту:
    /app - каталог с приложением (tg bot)
    /server - сервер/обработчик запросов от бота через FastAPI
    train.py - файл, коди из которого использовался для обучения модели
    Также dockerfile's для сборки контейнеров и развертывания



