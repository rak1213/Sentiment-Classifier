# Sentiment Classifier

## A00463237 Sentiment Classification Assignment

This project is focused on building and comparing multiple sentiment classification models. The goal is to accurately classify text data into predefined sentiment categories: Positive (2.0), Neutral (1.0), and Negative (0.0). This README documents the process and outcomes of the sentiment classification models developed for the assignment.

### Project Structure

The project utilizes Jupyter Notebooks (`*.ipynb`) to ensure that the process and results are transparent and easily understandable. The key components of the project include:

- **Data Preprocessing**: Scripts for cleaning and preparing the data for modeling.
- **Training**: Notebooks for training sentiment classification models along with utilization of Optuna for optimizing model performance.
- **Models**: Stores the saved models. 
- **Inference**: Scripts for applying the trained models to unseen data and evaluating their performance.
- **eda**: Notebook showing data analysis of the twitter dataset.
- **utils**: Scripts for saving and loading the models.
   
### Implementation Details
-- **Model Training Pipeline**:
The training process is encapsulated within a pipeline that automates the sequence of vectorization and classification. This approach ensures consistency and efficiency in transforming the raw text data into a format suitable for model training and subsequent predictions. The pipeline consists of:

A vectorization step that converts text data into numerical features. Depending on the model and the nature of the dataset, this could involve techniques like CountVectorizer, TF-IDFVectorizer, or word embeddings. For simplicaity we have taken TF-IDFVectorizer.

A classification step where the numerical features are used to train a sentiment classification model. Various classifiers are explored, including Logistic Regression, Naive Bayes, Support Vector Machines, and Random Forest Classifier.

-- **Hyperparameter Optimization with Optuna**:
Optuna is integrated into the model training process to systematically search for the best hyperparameters. This optimization not only enhances model accuracy but also contributes to understanding the impact of different hyperparameters on model performance. The best model configuration identified through Optuna's optimization trials is saved, allowing for easy reproduction of results and application to new datasets.

### Inference Results

The project evaluates three different models: Multinomial Naive Bayes, Random Forest, and Logistic Regression. Here are their performance metrics on the test dataset:

#### Performing Inference with MultinomialNB

        precision    recall  f1-score   support

     0.0       0.89      0.21      0.35      1001
     1.0       0.50      0.94      0.65      1430
     2.0       0.86      0.48      0.61      1103


Accuracy: 0.588568194680249


#### Performing Inference with Random Forest

          precision    recall  f1-score   support

     0.0       0.90      0.62      0.73      1001
     1.0       0.67      0.91      0.77      1430
     2.0       0.89      0.73      0.81      1103



Accuracy: 0.7705149971703452


#### Performing Inference with Logistic Regression

          precision    recall  f1-score   support

     0.0       0.79      0.64      0.71      1001
     1.0       0.67      0.82      0.74      1430
     2.0       0.83      0.74      0.78      1103


Accuracy: 0.743350311262026


### Conclusion

The Random Forest classifier achieved the highest accuracy among the tested models, followed closely by the Logistic Regression model. The Multinomial Naive Bayes model, while useful, did not perform as well as the other two in this task. These results highlight the importance of model selection and hyperparameter tuning in building effective text classification systems.

### Note

All project files are maintained in Jupyter Notebook format to visually present the results and ensure that the entire process, from data preprocessing to model inference, is transparent and reproducible.

