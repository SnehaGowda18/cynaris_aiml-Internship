# Week 4 Day 5 – Sentiment Classifier: Deploy & Document

## Description

This project implements a binary sentiment classification system using machine learning. A Logistic Regression model is trained on a text dataset and compared with a Random Forest classifier using accuracy, precision, recall, and a classification report. The project also generates a confusion matrix and ROC-AUC curve to evaluate model performance.

The trained Logistic Regression model is serialized using Joblib and deployed as a REST API with FastAPI. Users can send text through the `/predict` endpoint to receive a sentiment prediction, demonstrating an end-to-end machine learning deployment workflow.

## Tools & Technologies

* Python
* Scikit-learn
* Logistic Regression
* Random Forest Classifier
* TF-IDF Vectorizer
* Joblib
* FastAPI
* Uvicorn
* Matplotlib
* VS Code
* Git & GitHub
