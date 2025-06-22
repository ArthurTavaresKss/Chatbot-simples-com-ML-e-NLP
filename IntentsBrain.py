import tkinter as tk
from tkinter import scrolledtext
import json
import spacy
import unidecode
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Carregar modelo NLP
nlp = spacy.load("pt_core_news_sm")

# Carregar dados do JSON
with open("ChatbotMercado/intents.json", 'r', encoding='utf-8') as arquivo:
    dados = json.load(arquivo)

# Preparar dados
intents = dados['intents']
texts = []
labels = []

for intent in intents:
    for pattern in intent['patterns']:
        texts.append(unidecode.unidecode(pattern))
        labels.append(intent['name'])

# Lematizar
processed_texts = []
for text in texts:
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
    processed_texts.append(lemmatized if lemmatized.strip() else "placeholder")

# Vetorização e modelo
vectorizer = TfidfVectorizer(max_df=0.7, min_df=1)
X = vectorizer.fit_transform(processed_texts)

model = MultinomialNB(alpha=0.1)
model.fit(X, labels)


# Função de resposta
def generateResponse(intention, input_text):
    processed_input = unidecode.unidecode(input_text.lower())
    input_tokens = processed_input.split()

    for intent in intents:
        if intent['name'] == intention:
            matching_responses = []
            for response in intent['responses']:
                context = response.get('context', [])
                if not context:
                    matching_responses.append(response['text'])
                else:
                    for keyword in context:
                        if unidecode.unidecode(keyword.lower()) in input_tokens or keyword.lower() in processed_input:
                            matching_responses.append(response['text'])
                            break
            if matching_responses:
                return random.choice(matching_responses)
            return random.choice([r['text'] for r in intent['responses'] if not r.get('context')])
    
    return "Desculpe, não entendi. Pode repetir?"

# Função do botão
def enviar():
    user_input = entrada.get()
    if user_input.lower() == "sair":
        janela.destroy()
        return

    entrada.delete(0, tk.END)

    chat.insert(tk.END, f"Você: {user_input}\n")

    # Pré-processar
    doc = nlp(unidecode.unidecode(user_input))
    lemmatized = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
    lemmatized = lemmatized if lemmatized.strip() else "placeholder"
    vetor = vectorizer.transform([lemmatized])

    # Predição
    pred = model.predict(vetor)

    resposta = generateResponse(pred[0], user_input)

    chat.insert(tk.END, f"Bot: {resposta}\n\n")
    chat.see(tk.END)

# ---------------------------
# Interface Tkinter
# ---------------------------

janela = tk.Tk()
janela.title("Chatbot Mercado")
janela.geometry("500x550")
janela.resizable(False, False)

chat = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=60, height=25, font=("Arial", 10))
chat.pack(pady=10)
chat.insert(tk.END, "🤖 Chatbot iniciado! Digite 'sair' para encerrar.\n\n")
chat.configure(state='normal')

entrada = tk.Entry(janela, width=60, font=("Arial", 12))
entrada.pack(pady=5)

botao = tk.Button(janela, text="Enviar", command=enviar, bg="#4CAF50", fg="white", width=20)
botao.pack(pady=5)

janela.mainloop()
