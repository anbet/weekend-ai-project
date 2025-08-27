from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def tune_hyperparameters(X, y):
    """ Tune Hyperparameters for the sentiment analysis model """

    # Define parameter grid for GridSearchCV
    param_gird = {
        "logestic_regression": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__panalty": ['l1', 'l2'],
            "tfidf_max_features": [1000, 3000, 5000],
        },
        "nave_bayes": {
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
            "tfidf_max_features": [1000, 3000, 5000],
        },
    }

    best_models = {}
    for model_name, param_grid in param_gird.items():
        # Create a pipeline with TF-IDF and the model
        print(f"Tuning hyperparameters for {model_name}...")
        if model_name == "logestic_regression":
            model = LogisticRegression(solver='liblinear')
        else:
            model = MultinomialNB()
        
        pipeline = Pipeline([
            ('tidf', TfidfVectorizer(stop_words='english')),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)
        best_models[model_name] = {
            'model': grid_search.best_estimator_,
            'score': grid_search.best_score_,
            'params': grid_search.best_params_
        }
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")
        print("-" * 50)
        
    return best_models