# SMS Spam Classification using NLP



## Objective

To build and compare natural language processing (NLP) pipelines that automatically classify SMS text messages as either `ham` (legitimate) or `spam` using classical machine learning algorithms.

## Tools & Frameworks

* **Language & Runtime:** Python 3.x, Jupyter Notebook / Kaggle Kernel


* **Data Processing:** `pandas`, `numpy`

* **NLP & Vectorization:** `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`)


* **Machine Learning Models:** `scikit-learn` (`MultinomialNB`, `LogisticRegression`)


* **Evaluation Metrics:** `scikit-learn` (`accuracy_score`, `classification_report`, `train_test_split`)



## Dataset Overview

The dataset (`Spam_SMS.csv`) contains labeled SMS message instances:

* **Total Records:** 5,574 messages


* **Unique Messages:** 5,159


* **Class Distribution:** 4,827 `ham` messages and 747 `spam` messages


* **Columns:** `Class` (target label: `ham` or `spam`), `Message` (raw SMS text)



## Methodology

**1. CountVectorizer + Multinomial Naive Bayes**

* **Feature Extraction:** Transformed text into a Document-Term Matrix (DTM) using `CountVectorizer` with English stop-word removal, unigram/bigram extraction (`ngram_range=(1,2)`), and a minimum document frequency threshold of `min_df=0.005` (resulting in 294 features).


* **Train/Test Split:** Split data into an 80% training set and a 20% testing set (`test_size=0.2`, `random_state=42`).


* **Modeling:** Trained a `MultinomialNB` (Multinomial Naive Bayes) classifier.



**2. TF-IDF + Logistic Regression**

* **Feature Extraction:** Converted raw text into TF-IDF numerical vectors using `TfidfVectorizer` with English stop-word filtering, unigram/bigram tokenization (`ngram_range=(1,2)`), and a frequency threshold of `min_df=0.007` (resulting in 176 features).


* **Train/Test Split:** Split data into a 75% training set and a 25% testing set (`test_size=0.25`, `random_state=42`).


* **Modeling:** Trained a standard `LogisticRegression` model.



## Performance Results

                    precision    recall  f1-score   support

         ham           0.97      1.00      0.98      1203
        spam           0.98      0.81      0.89       191

    accuracy                               0.97      1394
    macro avg          0.98      0.90      0.94      1394
    weighted avg       0.97      0.97      0.97      1394

Accuracy: 0.9720229555236729
