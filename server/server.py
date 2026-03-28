from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from inference import SalaryPredictor


model = SalaryPredictor()

class Data(BaseModel):
    description: str = Field(
        ...,
        json_schema_extra={
            "example": "Требования: опыт работы от 3 лет, знание Python и ML. Обязанности: разработка моделей машинного обучения для анализа данных. Описание: компания занимается разработкой AI решений для бизнеса. Опыт: от 3 лет в сфере Data Science."
        },
    )


app = FastAPI()

@app.post('/predict')
def predict_salary(text: Data) -> dict[str, float]:
    '''
    Эндпоинт для получения предсказания вилки зарплаты по описанию вакансии.
    Принимает JSON с полем "description" и возвращает JSON с полями "salary_min" и "salary_max".
    '''
    prediction = model.predict(text.description)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)