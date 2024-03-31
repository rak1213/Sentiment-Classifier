# Sentiment Classifier

## A00463237 Sentiment Classification Assignment

This project is focused on building and comparing multiple sentiment classification models. The goal is to accurately classify text data into predefined sentiment categories: Positive (2.0), Neutral (1.0), and Negative (0.0). This README documents the process and outcomes of the sentiment classification models developed for the assignment.

### Project Structure

The project utilizes Jupyter Notebooks (`*.ipynb`) to ensure that the process and results are transparent and easily understandable. The key components of the project include:

- **Data Preprocessing**: Scripts for cleaning and preparing the data for modeling.
- **Model Training**: Notebooks for training sentiment classification models.
- **Hyperparameter Tuning**: Utilization of Optuna for optimizing model performance.
- **Inference**: Scripts for applying the trained models to unseen data and evaluating their performance.
- **eda**: Notebook showing data analysis of the twitter dataset.

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

