from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import unidecode
import random
import spacy
import json

# Carregar o modelo de linguagem
nlp = spacy.load("pt_core_news_sm")
model = MultinomialNB(alpha=0.1)

# Carregar os dados do JSON
with open("ChatbotMercado/intents.json", 'r', encoding='utf-8') as intRead:
    dados = json.load(intRead)

# Definir as intenções, nomes e patterns
intents = dados['intents']
texts = []
names = []

for intent in intents:
    for pattern in intent['patterns']:
        texts.append(unidecode.unidecode(pattern))
        names.append(intent['name'])

# Lematizar os textos
processed_texts = []
for text in texts:
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
    processed_texts.append(lemmatized if lemmatized.strip() else "placeholder")

# Vetorizar para TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7, min_df=1)
X = vectorizer.fit_transform(processed_texts)

# Separar dados para treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, names, 
    test_size=0.2, 
    random_state=48402, 
    stratify=names
    )

# Treinar
model.fit(X_train, Y_train)

# Função para gerar resposta
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
    
    return "Desculpe, não entendi. Pode repetir com outras palavras?"

# Chatbot
text_to_verify = ""
while text_to_verify.lower() != "sair":
    text_to_verify = input(f"\033[91m[Digite uma mensagem ou 'sair' para cancelar]\033[0m\n\033[93m>\033[94m ")
    if text_to_verify.lower() == "sair":
        print("\n\033[91mEncerrando...\033[0m\n")
        break

    text_to_verify = unidecode.unidecode(text_to_verify)
    text_to_verify_doc = nlp(text_to_verify)
    text_to_verify_lemmatized = ' '.join([token.lemma_.lower() for token in text_to_verify_doc if not token.is_stop and not token.is_punct])
    text_to_verify_lemmatized = text_to_verify_lemmatized if text_to_verify_lemmatized.strip() else "placeholder"

    text_to_verify_vectorized = vectorizer.transform([text_to_verify_lemmatized])
    prediction_verify = model.predict(text_to_verify_vectorized)

    response = generateResponse(prediction_verify, text_to_verify)
    print(f"\033[93m--> \033[94m{response}\033[0m\n")

# Avaliar modelo
scores = cross_val_score(model, X, names, cv=5, scoring='accuracy')
Y_predicted = model.predict(X_test)

print(f"\033[95mAcurácia média: {scores.mean():.2f}")
print(f"\033[95mAcurácias individuais: {scores}\n")
print(f"\033[95m{classification_report(Y_test, Y_predicted)}")