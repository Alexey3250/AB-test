import os
from catboost import CatBoostClassifier, CatBoost
import pandas as pd
from sqlalchemy import create_engine
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
import pydantic
from tqdm import tqdm
from typing import List


'''
ФУНКЦИИ ПО ЗАГРУЗКЕ МОДЕЛЕЙ
'''
# Проверка если код выполняется в лмс, или локально
def get_model_path(model_name: str) -> str:
    """Просьба не менять этот код"""
    if model_name not in ["test", "control"]:
        raise ValueError("model_name should be either 'test' or 'control'")
        
    if os.environ.get("IS_LMS") == "1":
        return f'/workdir/user_input/model_{model_name}'
    else:
        # Это ваш локальный путь к моделям
        paths = {
            "test": "C:/Users/Alex/AB-test/final_with_DL/model_A.cbm",
            "control": "C:/Users/Alex/AB-test/final_with_DL/model_B.cbm"
        }
        return paths[model_name]

class CatBoostWrapper(CatBoost):
    def predict_proba(self, X):
        return self.predict(X, prediction_type='Probability')

# Загрузка модели
def load_model(model_name: str):
    model_path = get_model_path(model_name)
    model = CatBoostWrapper()
    model.load_model(model_path)
    return model

# Загрузка моделей A и B (test и control)
model_A = load_model('test')
model_B = load_model('control')
print("Модели загружены")



'''
Получение данных из базы данных
'''
# Определяем функцию для получения данных из базы данных PostgreSQL
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    total_rows_for_5_percent = 3844565
    # total_rows_for_10_percent = 7689263  # total number of rows in your dataset

    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)

    chunks = []
    with tqdm(total=total_rows_for_5_percent, desc="Loading data") as pbar:
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
            chunks.append(chunk_dataframe)
            pbar.update(CHUNKSIZE)

    conn.close()

    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    query = "alexey_efimik_data_ready_wv3"
    return batch_load_sql(query)

features = load_features()
print("Данные загружены")

def predict_posts(user_id: int, model, limit: int):
    # Фильтруем записи, относящиеся к конкретному user_id
    user_features = features[features.user_id == user_id]

    # Вычисляем вероятности для каждого post_id для конкретного user_id
    user_features['probas'] = model.predict_proba(user_features.drop('user_id', axis=1))[:, 1]

    # Сортируем DataFrame по 'probas' в порядке убывания и получаем первые 'limit' записей
    top_posts = user_features.sort_values('probas', ascending=False).iloc[:limit]

    # Возвращаем 'post_id' лучших записей в виде списка
    return top_posts['post_id'].tolist()


def load_post_texts_df():
    global post_texts_df
    print("Загружаю все тексты постов...")
    query = "SELECT * FROM post_text_df"
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    post_texts_df = pd.read_sql(query, con=engine)
    print("Все тексты постов успешно загружены в память.")


def load_post_texts(post_ids: List[int]) -> List[dict]:
    global post_texts_df
    if post_texts_df is None:
        raise ValueError("Таблица с текстами постов не загружена. Сначала вызовите функцию load_post_texts_df().")

    # Извлекаем записи из памяти
    records_df = post_texts_df[post_texts_df['post_id'].isin(post_ids)]
    return records_df.to_dict("records")


'''
ЗАГРУЗКА МОДЕЛЕЙ И ФИЧЕЙ (БЕЗ ПОТОКОВ)
'''

# Глобальная переменная для хранения данных
post_texts_df = None
load_post_texts_df()

'''
БЛОКИ ПО АБ-ТЕСТИРОВАНИЮ
'''

'''
Функция для разделения пользователей на группы для АБ-тестирования
'''
import hashlib

SALT = 'some_salt'  # Соль для хэширования
SPLIT_RATIO = 0.5  # Процентное соотношение разбиения

def get_exp_group(user_id: int) -> str:
    salted_user_id = str(user_id) + SALT
    user_hash = hashlib.md5(salted_user_id.encode())
    user_hash_value = int(user_hash.hexdigest(), 16)  # Преобразование хэша в целое число
    if user_hash_value % (10**16) / (10**16) < SPLIT_RATIO:
        return 'control'
    else:
        return 'test'

'''
 Реализация функций для применения моделей
'''
def predict_posts_with_control_model(user_id: int, limit: int):
    return predict_posts(user_id, model_A, limit)

def predict_posts_with_test_model(user_id: int, limit: int):
    return predict_posts(user_id, model_B, limit)


'''
FASTAPI
'''
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


app = FastAPI()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    
    # Определяем группу A/B-тестирования для данного пользователя
    exp_group = get_exp_group(id)

    # Применяем соответствующую модель в зависимости от группы
    if exp_group == 'control':
        post_ids = predict_posts_with_control_model(id, limit)
    elif exp_group == 'test':
        post_ids = predict_posts_with_test_model(id, limit)
    else:
        raise ValueError('unknown group')

    # Загружаем тексты постов для рекомендаций
    records = load_post_texts(post_ids)

    # Формируем ответ
    posts = []
    for rec in records:
        rec["id"] = rec.pop("post_id")
        try:
            posts.append(PostGet(**rec))
        except pydantic.error_wrappers.ValidationError as e:
            print(f"Validation error for record {rec}: {e}")
    return posts
