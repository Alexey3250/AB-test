import requests
from datetime import datetime
import random
import json
from tqdm import tqdm

# Конфигурация запросов
URL = "http://127.0.0.1:8000/post/recommendations/"
TIME = datetime.now().isoformat()
LIMIT = 5

# Функция для выполнения одного запроса и получения статистики
def make_request(user_id):
    response = requests.get(URL, params={"id": user_id, "time": TIME, "limit": LIMIT})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка в запросе для user_id={user_id}: {response.text}")
        return None

# Собираем статистику для 1000 случайных user_id
statistics = {
    "test": 0,
    "control": 0,
    "errors": 0
}

# Повторяем запросы 100 раз с разными user_ids
for user_id in tqdm(range(100)):
    user_id = random.randint(200, 168552) # Предположим, что user_id варьируются от 1 до 10^6
    result = make_request(user_id)
    if result:
        group = result["exp_group"]
        statistics[group] += 1
    else:
        statistics["errors"] += 1

# Выводим результаты статистики
print(json.dumps(statistics, indent=4))
