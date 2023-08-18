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
import hashlib


'''
Функция для определения группы A/B теста
'''
SALT = "SomeRandomSaltValue"  # соль для хеширования
THRESHOLD = 2**32 // 2  # пороговое значение для разбиения на группы

def get_exp_group(user_id: int) -> str:
    # Конвертировать user_id в строку и добавить соль
    input_str = str(user_id) + SALT
    
    # Получить MD5-хеш
    hashed = hashlib.md5(input_str.encode()).hexdigest()
    
    # Преобразовать первые 8 символов хеша (это 32 бита) в число
    int_value = int(hashed[:8], 16)
    
    # Определить группу на основе полученного числа
    if int_value < THRESHOLD:
        return "control"
    else:
        return "test"



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
            "test": "/Users/ilya/Desktop/GitHub_Repositories/DataSets/DataSets for Final Project/catboost_model_data10_best_hitrate.cbm",
            "control": "/Users/ilya/Desktop/GitHub_Repositories/DataSets/DataSets for Final Project/catboost_model_data10_lower_hitrate.cbm"
        }
        return paths[model_name]


class CatBoostWrapper(CatBoost):
    def predict_proba(self, X):
        return self.predict(X, prediction_type='Probability')

# Загрузка модели
def load_model(model_name: str):
    """
    Загрузка одной из моделей: тестовой или контрольной
    """
    model_path = get_model_path(model_name)
    
    model = CatBoostWrapper()
    model.load_model(model_path)
    
    return model



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
    query = "ilia_svetlichnyi_features_lesson_22_5_percent"
    return batch_load_sql(query)



def predict_posts(user_id: int, chosen_model, limit: int):
    # Фильтруем записи, относящиеся к конкретному user_id
    user_features = features[features.user_id == user_id]

    # Вычисляем вероятности для каждого post_id для конкретного user_id
    user_features['probas'] = chosen_model.predict_proba(user_features.drop('user_id', axis=1))[:, 1]

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


features = load_features()
print("Данные загружены")
# Глобальная переменная для хранения данных
post_texts_df = None
load_post_texts_df()


'''
FASTAPI
'''
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


app = FastAPI()

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        user_id: int,
        time: datetime,
        limit: int = 5
) -> Response:
    
    # Определение группы для данного пользователя
    exp_group = get_exp_group(user_id)
    
    # Выбор модели в зависимости от группы
    chosen_model = load_model(exp_group)
    
    # Тут идет ваш код для предсказания, используя chosen_model
    post_ids = predict_posts(user_id, chosen_model, limit)
    
    # Загрузка текстов постов
    records = load_post_texts(post_ids)

    # Создание списка рекомендованных постов
    posts = []
    for rec in records:
        rec["id"] = rec.pop("post_id")
        try:
            posts.append(PostGet(**rec))
        except pydantic.error_wrappers.ValidationError as e:
            print(f"Validation error for record {rec}: {e}")
    
    # Возвращаем список рекомендаций с указанием группы
    return Response(exp_group=exp_group, recommendations=posts)
