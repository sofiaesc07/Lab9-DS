# Librerias generales
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# --- Dash ---
import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# --- Analisis de sentimientos ---
from nltk.sentiment import SentimentIntensityAnalyzer

import chart_studio
chart_studio.tools.set_config_file(world_readable=True, sharing='public')


data = pd.read_csv('train.csv')

# Combinar keyword y location en un solo campo
data['features'] = data['keyword'].fillna('') + ' ' + data['location'].fillna('') + ' ' + data['text']

# Convertir a minúsculas
data['text'] = data['text'].astype(str).str.lower()

# Quitar caracteres especiales, URL y números
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'http\S+', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'\d+', '', x))

# Quitar stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Dividir los datos en tweets de desastres y no desastres
disaster_tweets = data[data['target'] == 1]['text']
non_disaster_tweets = data[data['target'] == 0]['text']

# Tokenización y conteo de palabras
disaster_word_counts = Counter(' '.join(disaster_tweets).split())
non_disaster_word_counts = Counter(' '.join(non_disaster_tweets).split())

# Crear vectorizador para unigramas y bigramas
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000000)  

# Obtener la matriz de términos de documento
X = vectorizer.fit_transform(data['text'])

# Obtener las palabras más comunes en unigramas y bigramas para cada clase
disaster_common_words = vectorizer.get_feature_names_out()
non_disaster_common_words = vectorizer.get_feature_names_out()

# Nube de palabras de desastres
wordcloud_disaster = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(disaster_word_counts)
# Nube de palabras para tweets no desastres
wordcloud_non_disaster = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate_from_frequencies(non_disaster_word_counts)

data['tweet_length'] = data['text'].apply(len)

common_words = set(disaster_word_counts.keys()) & set(non_disaster_word_counts.keys())

# Crear el analizador de sentimiento
sia = SentimentIntensityAnalyzer()

# Calcular los valores de sentimiento para cada texto en tus datos
sentimiento = np.zeros(len(data['text']))
for k in range(len(data['text'])):
    sentimiento[k] = sia.polarity_scores(data['text'][k]).get('compound')

data['sentimiento'] = sentimiento

# Calcular la negatividad para cada tweet y agregarla al DataFrame
negatividad = []
for text in data['text']:
    sentiment_scores = sia.polarity_scores(text)
    negatividad.append(sentiment_scores['neg'])

data['negatividad'] = negatividad

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[['text', 'negatividad']], data['target'], random_state=41)

# Crear el modelo con TF-IDF y Naive Bayes
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Puedes ajustar el número máximo de características
    ('clf', MultinomialNB()),
])

# Crear el modelo con TF-IDF y SVM
text_svm = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('clf', SVC(kernel='linear')),
])

# Crear el modelo con TF-IDF y Random Forest Classifier
text_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('clf', RandomForestClassifier(n_estimators=100, random_state=0)),
])

# Entrenar los modelos
text_clf.fit(X_train['text'], y_train)
text_svm.fit(X_train['text'], y_train)
text_rf.fit(X_train['text'], y_train)

# Evaluar los modelos
y_pred = text_clf.predict(X_test['text'])
accuracy = accuracy_score(y_test, y_pred)

y_pred_svm = text_svm.predict(X_test['text'])
accuracy_svm = accuracy_score(y_test, y_pred_svm)

y_pred_rf = text_rf.predict(X_test['text'])
accuracy_rf = accuracy_score(y_test, y_pred_rf)

##funcion para clasificar
def classify_tweet(tweet):
    prediction = text_clf.predict([tweet])
    if prediction[0] == 1:
        return "Desastre"
    else:
        return "No desastre"

    
# Calcular la media del sentimiento para los tweets de desastre y no desastre
disaster_sentiment_mean = data[data['target'] == 1]['sentimiento'].mean()
non_disaster_sentiment_mean = data[data['target'] == 0]['sentimiento'].mean()

# Calcular la cantidad de tweets clasificados como desastres y no desastres
disaster_tweets = sum(data['target'] == 1)
non_disaster_tweets = sum(data['target'] == 0)


###########################################################################################3

###################################################################################
# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1("Clasificador de Tweets"),
    html.H2("Explora los datos"),
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': 'Tweets de desastres', 'value': 'disaster'},
            {'label': 'Tweets no desastres', 'value': 'non_disaster'}
        ],
        value='disaster'
    ),
    dcc.Graph(id='data-graph'),
    
    # Nube de palabras
    html.Div([
        html.H2("Nube de Palabras"),
        dcc.Dropdown(
            id='wordcloud-dropdown',
            options=[
                {'label': 'Tweets de desastres', 'value': 'disaster'},
                {'label': 'Tweets no desastres', 'value': 'non_disaster'}
            ],
            value='disaster'
        ),
        dcc.Graph(id='wordcloud-graph'),
    ]),
    
    # Histograma de la longitud de Tweets
    html.Div([
        html.H2("Distribución de Longitud de Tweets"),
        dcc.Graph(
            figure={
                'data': [
                    {
                        'x': data[data['target'] == 1]['tweet_length'],
                        'type': 'histogram',
                        'name': 'Desastre',
                        'opacity': 0.7,
                        'marker': {'color': 'tomato'}
                    },
                    {
                        'x': data[data['target'] == 0]['tweet_length'],
                        'type': 'histogram',
                        'name': 'No Desastre',
                        'opacity': 0.7,
                        'marker': {'color': 'teal'}
                    }
                ],
                'layout': {
                    'xaxis': {'title': "Longitud del Tweet"},
                    'yaxis': {'title': "Frecuencia"},
                    'bargap': 0.2,
                    'bargroupgap': 0.1,
                    "title": "Distribución de Longitud de Tweets"
                }
            },
            style={'width': "100%"}
        ),
    ]),
       
    # Gráfico de barras para la media del sentimiento
    html.Div([
        html.H2("Media del Sentimiento"),
        dcc.Graph(
            id='sentiment-mean-graph',
            figure={
                'data': [
                    {
                        'x': ['Desastre', 'No Desastre'],
                        'y': [disaster_sentiment_mean, non_disaster_sentiment_mean],
                        'type': 'bar',
                        'name': 'Media del Sentimiento',
                        'marker': {'color': ['tomato', 'teal']}
                    }
                ],
                'layout': {
                    'xaxis': {'title': "Tipo de Tweet"},
                    'yaxis': {'title': "Media del Sentimiento"},
                    "title": "Media del Sentimiento para Tweets de Desastre y No Desastre"
                }
            },
            style={'width': "100%"}
        ),
    ]),
    
    
    # Gráfico de pie para la clasificación de tweets
    html.Div([
        html.H2("Distribución de Clasificación de Tweets"),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "labels": ["Desastre", "No Desastre"],
                        "values": [disaster_tweets, non_disaster_tweets],
                        "type": "pie",
                        "name": "Clasificación",
                        "marker": {"colors": ["tomato", "teal"]},
                        "textinfo": "label+percent",
                        "insidetextorientation": "radial"
                    }
                ],
                "layout": {
                    "title": {"text": "Distribución de Clasificación de Tweets"}
                }
            },
            style={'width': "100%"}
        ),
    ]),
    
    # Clasificación de tweets
    html.Div([
        html.H2("Selecciona un modelo para la clasificación"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': "Naive Bayes", "value": "nb"},
                {'label': "SVM", "value": "svm"},
                {'label': "Random Forest", "value": "rf"}
            ],
            value="nb"
        ),
        dcc.Input(id="tweet-input", type="text", placeholder="Introduce un tweet para clasificar"),
        html.Button("Clasificar", id="classify-button", n_clicks=0),
        html.Div(id="classification-output")
    ])
])


# Definir la función de callback para la exploración de datos
@app.callback(
    Output('data-graph', 'figure'),
    [Input('data-dropdown', 'value')]
)


def update_graph(value):
    if value == 'disaster':
        word_counts = disaster_word_counts
        title = 'Palabras más comunes en tweets de desastres'
        color = 'tomato'  # Color para los tweets de desastres
    else:
        word_counts = non_disaster_word_counts
        title = 'Palabras más comunes en tweets no desastres'
        color = 'teal'  # Color para los tweets no desastres

    data = go.Bar(
        x=list(word_counts.keys()),
        y=list(word_counts.values()),
        marker_color=color  # Aplicar el color seleccionado
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Palabras'),
        yaxis=dict(title='Frecuencia')
    )

    return go.Figure(data=data, layout=layout)


# Definir la función de callback para la nube de palabras
@app.callback(
    Output('wordcloud-graph', "figure"),
    [Input('wordcloud-dropdown', "value")]
)
def update_wordcloud(value):
    if value == "disaster":
        img = wordcloud_disaster.to_image()
    else:
        img = wordcloud_non_disaster.to_image()

    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig

# Definir la función de callback para la clasificación de tweets
@app.callback(
    Output('classification-output', 'children'),
    [Input('classify-button', 'n_clicks')],
    [State('tweet-input', 'value'), State('model-dropdown', 'value')]
)
def classify_tweet(n_clicks, tweet, model):
    if n_clicks > 0:
        if model == 'nb':
            prediction = text_clf.predict([tweet])
        elif model == 'svm':
            prediction = text_svm.predict([tweet])
        elif model == 'rf':
            prediction = text_rf.predict([tweet])
        
        return f"{'Desastre' if prediction[0] == 1 else 'No desastre'}"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)