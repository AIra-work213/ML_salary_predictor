from transformers import BertTokenizer, BertModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import llamacpp


tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

class BertRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(768, 2),
            torch.nn.ReLU()
        )
    def forward(self, x):
        x = self.bert(x).last_hidden_state[:, 0, :]
        x = self.ffnn(x)
        return x

model = BertRegressor()

model.load_state_dict(torch.load('model_weights (6).pth'))
model.eval()

class RequestData(BaseModel):
    title: str = Field(..., description="Название вакансии")
    experience: str = Field(..., description="Опыт, требуемый работодателем, 1 предложение")
    job_type: str = Field(..., description="Тип работы")
    description: str = Field(..., description="Полное описание вакансии, весь текст")
    key_skills: str = Field(..., description="Ключевые навыки, требуемый работодателем")
    company: str = Field(..., description="Полное название компании, 1 предложение")
    location: str = Field(..., description="Локация, где надо будет работать")
    date_of_post: str = Field(..., description="Дата публикации")
    type: str = Field(..., description="Закрытая или открытая вакансия")

class Data(BaseModel):
    description: str


def parser_text(desc: RequestData) -> str:
    titles_ru = ['Должность','Опыт','Тип работы','Описание','Ключевые навыки','Компания','Местоположение','Дата публикации','Тип']
    k=0
    s=""
    for field_name in desc.model_fields.keys():
        curr_field = getattr(desc, field_name)
        s += f'{titles_ru[k]}: {curr_field} '
        k += 1
    return s

def parser_answer(tensor) -> str:
    return f'Зарплатная вилка: {int(torch.min(tensor))} - {int(torch.max(tensor))}'


app = FastAPI()

@app.post('/')
def get_description(text: Data) -> str:
    with torch.no_grad():
        parser = PydanticOutputParser(
            pydantic_object=RequestData
        )
        template = """Преобразуй входные данные, строго следуя данному шаблону, если данные отсутствуют, то поставь знак минуса в соответствующем поле\n
        {format_instructions}\n
        Входные данные: {input_data}
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['input_data'],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        model_path = './'
        llm = llamacpp(
        model_path=model_path,
        n_ctx=2048, 
        n_threads=6,  
        n_gpu_layers=0,
        temperature=0.1,
        verbose=False
        )

        chain = prompt | llm | parser

        return parser_answer(model.predict(parser_text(chain.invoke({'input_data': Data.description}))))
