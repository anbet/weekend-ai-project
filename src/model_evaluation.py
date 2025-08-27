import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_data):
    """ Evaluate the model using test data """
    predictions = []
    actuals = []

    for text, sentiment in test_data:
        pred = model.predict_sentimental_local(text)
        predictions.append(pred['sentiment'])
        actuals.append(sentiment)

    # Generate confusion matrix
    cm = confusion_matrix(actuals, predictions, labels=model.best_model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.best_model.classes_, yticklabels=model.best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Sentiment analysis Confusion Matrix')
    plt.show()

    # Classification report
    report = classification_report(actuals, predictions, target_names=model.best_model.classes_)
    print("Classification Report:\n", report)
    return cm, report
        