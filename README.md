# Chatbot Mercado — Python + IA + NLP

Um chatbot inteligente, desenvolvido em Python, que reconhece intenções e responde de forma contextual utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Machine Learning**.

## Funcionalidades
- Reconhecimento de intenções a partir de texto.
- Geração de respostas personalizadas com contexto.
- Treinamento com aprendizado supervisionado (Naive Bayes + TF-IDF).
- Avaliação de desempenho com Cross-Validation e relatórios de classificação.
- Processamento de linguagem com lematização, remoção de acentos, stopwords e pontuação.

------

## Estrutura dos Arquivos
ChatbotMercado/

├── **intents.json** # Arquivo com as intenções, padrões e respostas

├── **IntentsBrain.py** # Código Python do chatbot

├── **requirements.txt** # Dependências do projeto

**README.md** # Este arquivo

------

## Tecnologias e Bibliotecas
- [*Python*](https://www.python.org/)
- [*scikit-learn*](https://scikit-learn.org/stable/) — Machine Learning
- [*spaCy*](https://spacy.io/) — Processamento de Linguagem Natural
- [*TfidfVectorizer*](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — Vetorização de texto
- [*Unidecode*](https://pypi.org/project/Unidecode/) — Remoção de acentos
- [*NumPy*](https://numpy.org/) — Suporte matemático
- [*JSON*](https://www.json.org/json-pt.html) — Banco de dados de intenções

------

## Instalação

1. Clone este repositório:
```bash
git clone https://github.com/ArthurTavaresKss/Chatbot-simples-com-ML-e-NLP.git
cd ChatbotMercado
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

3. Execute o chatbot:
```bash
python IntentsBrain.py
```

## Como funciona?

- O chatbot lê o arquivo intents.json contendo:
- patterns → Frases de treinamento.
- name → Nome da intenção.
- responses → Respostas possíveis (com ou sem contexto).
- Processa as frases (remoção de acentos, stopwords e lematização).
- Transforma os textos em vetores numéricos usando TF-IDF.
- Treina um modelo Naive Bayes para classificar intenções.
- Interage com o usuário, identifica a intenção e responde de acordo.

##  Exemplo do intents.json

```json
{
  "intents": [
    {
      "name": "Pedido",
      "patterns": [
        "Quero comprar arroz",
        "Vocês têm pão?",
        "Gostaria de solicitar um produto"
      ],
      "responses": [
        {"text": "Claro! Qual produto você deseja?", "context": ["comprar", "produto"]},
        {"text": "Perfeito, podemos verificar o estoque para você!"}
      ]
    }
  ]
}
```

###  Desenvolvedor
👤 Arthur Tavares
🚀 Projeto desenvolvido para fins educacionais e para aprendizado.

---


