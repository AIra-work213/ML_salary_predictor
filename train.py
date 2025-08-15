from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from tqdm import tqdm
import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('train.csv', on_bad_lines='skip', sep=';')
        self.df.drop(columns = ['№', 'id'], inplace=True)
        Ru_titles = {
            'title': 'Должность',
            'salary': 'Зарплата',
            'experience': 'Опыт',
            'job_type': 'Тип работы',
            'description': 'Описание',
            'key_skills': 'Ключевые навыки',
            'company': 'Компания',
            'location': 'Местоположение',
            'date_of_post': 'Дата публикации',
            'type': 'Тип'
        }

        self.df.rename(columns = Ru_titles, inplace=True)
        self.df = self.df[self.df['Зарплата'] != 'з/п не указана']

        self.targets = self.df['Зарплата']
        self.df = self.df.drop(columns = ['Зарплата'])
        self.columns_name = self.df.columns.to_list()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.df.iloc[idx].to_list()
        text = ''
        for i, j in zip(self.columns_name, s):
            text += f'{i}: {j}'

        y = self.targets.iloc[idx]
        y = self.clean_target(y)
        return (text, y)

    def clean_target(self, s, digits = set('0123456789')) -> list:
        # Проверяем, что s - строка, если нет - конвертируем
        if not isinstance(s, str):
            s = str(s)
        
        target = ''
        for i in range(len(s)):
            if s[i] in digits:
                target += s[i]
            else:
                target += ' '
        
        numbers = target.split()
        if not numbers:
            return [0, 0]
        
        # Всегда возвращаем список из 2 элементов
        result = list(map(int, numbers))
        if len(result) == 1:
            result.append(result[0])
        
        return result

data = Dataset()
dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, drop_last=True)



tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', local_files_only=True)

class BertRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased', local_files_only=True)
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2)
        )
    def forward(self, x):
        x = self.bert(x).last_hidden_state[:, 0, :]
        x = self.ffnn(x)
        return x

model = BertRegressor()

for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(3):
    for x, y in tqdm(dataloader, desc=f'Epoch {epoch+1}/3'):
        vectors = tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors='pt')

        optimizer.zero_grad()
        y_pred = model(vectors['input_ids'])

        y = torch.stack([torch.tensor(batch_item, dtype=torch.float32) for batch_item in y])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "model_weights.pth")

print('Готово')