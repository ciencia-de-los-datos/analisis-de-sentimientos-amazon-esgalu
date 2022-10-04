"""
Análisis de Sentimientos usando Naive Bayes
-----------------------------------------------------------------------------------------

El archivo `amazon_cells_labelled.txt` contiene una serie de comentarios sobre productos
de la tienda de amazon, los cuales están etiquetados como positivos (=1) o negativos (=0)
o indterminados (=NULL). En este taller se construirá un modelo de clasificación usando
Naive Bayes para determinar el sentimiento de un comentario.

"""

import numpy as np
import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    """
    df = pd.read_csv(
        "amazon_cells_labelled.tsv",
        sep="\t",
        header=None,
        names=["msg", "lbl"]
    )

    df_tagged = df[df["lbl"].notnull()]
    df_untagged = df[df["lbl"].isnull()]

    x_tagged = df_tagged["msg"]
    y_tagged = df_tagged["lbl"]

    x_untagged = df_untagged["msg"]
    y_untagged = df_untagged["lbl"]

    return x_tagged, y_tagged, x_untagged, y_untagged


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    """
    from sklearn.model_selection import train_test_split

    x_tagged, y_tagged, x_untagged, y_untagged = pregunta_01()

    x_train, x_test, y_train, y_test = train_test_split(
        x_tagged,
        y_tagged,
        test_size=0.1,
        random_state=12345,
    )

    return x_train, x_test, y_train, y_test


def pregunta_03():
    """
    Construcción de un analizador de palabras
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()

    analyzer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        lowercase=True
    ).build_analyzer()

    return lambda x: (stemmer.stem(w) for w in analyzer(x))


def pregunta_04():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import BernoulliNB

    x_train, x_test, y_train, y_test = pregunta_02()

    analyzer_ = pregunta_03()

    countVectorizer = CountVectorizer(
        analyzer=analyzer_,
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        binary=True,
        max_df=1.0,
        min_df=5
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("countVectorizer", countVectorizer),
            ("bernoulli", BernoulliNB()),
        ],
    )

    # Diccionario de parámetros para el GridSearchCV. 0.1 - 1.01 paso 0.1
    param_grid = {
        "bernoulli__alpha": np.arange(0.1, 1.01, 0.1),
    }

    # Instancia de GridSearchCV con el pipeline
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        refit=True,
        return_train_score=True,
    )

    gridSearchCV.fit(x_train, y_train)

    return gridSearchCV


def pregunta_05():
    """
    Evaluación del modelo
    """
    from sklearn.metrics import confusion_matrix

    gridSearchCV = pregunta_04()

    x_train, x_test, y_train, y_test = pregunta_02()

    # Matriz de confusion.
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=gridSearchCV.predict(x_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=gridSearchCV.predict(x_test),
    )

    return cfm_train, cfm_test


def pregunta_06():
    """
    Pronóstico
    """
    gridSearchCV = pregunta_04()

    x_tagged, y_tagged, x_untagged, y_untagged = pregunta_01()

    # Pronostico de polaridad del sentimiento para los datos no etiquetados
    y_untagged_pred = gridSearchCV.predict(x_untagged)

    return y_untagged_pred
