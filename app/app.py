from telebot import TeleBot
import os
import requests

BOT_TOKEN = os.getenv('BOT_TOKEN')

app = TeleBot(BOT_TOKEN)

@app.message_handler(['start'])
def helper(message):
    app.send_message(message.chat.id, "Привет, я бот, способный по описанию вакансии определить возможную зарплатную вилку")
    
@app.message_handler(['predict'])
def helper(message):
    app.send_message(message.chat.id, "Укажите все описание вакансии, указанное работодателем (в т.ч. дата публикации, опыт, название и т.п.)")
    text = app.register_next_step_handler(message, request)

def request(message):
    text = message.text
    response = requests.post('https://localhost:8000', json=text)
    app.send_message(message.chat.id, f"{response}")
