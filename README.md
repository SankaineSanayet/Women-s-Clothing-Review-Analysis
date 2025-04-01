# Women's Clothing E-Commerce Reviews Sentiment Analysis
This project is part of the Moringa Phase 4 final project.

# Overview
In the fast-growing e-commerce industry, customer reviews play a crucial role in shaping brand perception, influencing purchases and identifying areas for improvement. This project focuses on analyzing women’s clothing reviews to extract meaningful insights using Natural Language Processing (NLP) and data science techniques.

By leveraging sentiment analysis, text preprocessing and machine learning, we aim to classify customer sentiment, uncover key themes and provide actionable insights that businesses can use to enhance customer experience.

# Business Understanding
The primary objective of this analysis is to leverage Natural Language Processing (NLP) techniques to perform sentiment analysis on women’s e-commerce clothing reviews. This will help businesses extract valuable insights from customer feedback and enhance decision-making.

**Refined Objectives:**  
- Sentiment Classification: Build a model to classify customer reviews as Positive or Negative, using review text as the main feature. The goal is to achieve at least 80% accuracy and an F1-score of 0.80.
- Topic Modeling: Identify key themes in customer reviews (e.g., product quality, sizing, shipping experience) using techniques like Latent Dirichlet Allocation (LDA).
- Actionable Insights: Use visualizations (e.g., word clouds, sentiment trends) to provide recommendations for product improvements and customer engagement strategies.

**Target Audience**
This analysis is valuable for:
- E-commerce Businesses – To understand customer sentiment and improve offerings.
- Product Managers & Merchandisers – To make data-backed product decisions.
- Marketing Teams – To craft targeted campaigns based on customer preferences.
- Data Scientists & Analysts – To explore NLP techniques in a real-world business context.

**Success Metrics**
- Achieve at least 80% accuracy and an F1-score of 0.80 on the test set.
- Identify key factors like product quality, fit, pricing and customer service.
- Provide clear, actionable recommendations based on sentiment trends and topic distributions.

# Data Understanding
The dataset used in this analysis consists of customer reviews for women’s clothing, sourced from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews). This dataset contains 23,486 rows and 10 features. It contains text-based feedback along with numerical ratings and other metadata.

  **Key Features**
  - Review Text: The primary textual feedback provided by customers.
  - Rating: A numerical score (1-5) reflecting customer satisfaction.
  - Recommended IND: A binary indicator of whether a customer recommends the product (1 = Yes, 0 = No).
  - Positive Feedback Count: The number of customers who found a review helpful.
  - Age: The age of the reviewer.
    
  **Target Variable**
    - 
# Data Preparation
Before analysis, the dataset was prepared by handling missing values, duplicates, and necessary transformations:
- Handling Missing Data: 'Reviews','Division Name', 'Department Name' and 'Class Name' with missing text were removed, as they provide no value for sentiment analysis.
- Dropping irrelevant columns: 'Clothing ID','Title','Unnamed: 0' were dropped as they are not essential for analysis. 
- Removing Duplicates: Duplicate reviews were identified and dropped to ensure unbiased analysis.
- Sentiment Labeling: A new Sentiment column was created based on the Recommended ID: Positive: 1, Negative: 0

# Data Preprocessing and Cleaning(Text Specific)
To prepare the text for sentiment analysis, several preprocessing techniques were applied to ensure the text data was structured and optimized for feature extraction and model training:
- Text Cleaning: Removed URLs, HTML tags, special characters, extra spaces, and numbers to retain only meaningful words.
- Lowercasing: Standardized all text to lowercase for consistency.
- Stopword Removal: Eliminated common words (e.g., “the”, “and”) that do not contribute to sentiment.
- Normalization: Limited elongated words (e.g., “sooo” → “soo”) and allowed up to two consecutive repetitions.
- Tokenization: Split text into individual words (tokens) for further processing.
- Lemmatization: Reduced words to their base forms (e.g., “running” → “run”) to maintain word consistency.
- Ngram Generation: Created bigrams to capture context and relationships between words.

# Feature Engineering and Representation
