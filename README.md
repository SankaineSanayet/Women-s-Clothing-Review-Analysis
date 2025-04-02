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
To prepare the text data for machine learning models, we transformed the cleaned review text into numerical features using Text Vectorization and Dimensionality Reduction techniques.

1. Text Vectorization
- Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical values.
- Incorporated unigrams (single words) and bigrams (word pairs) to capture word meaning and context.
- Resulting feature matrix: (22625, 5000) features.

2. Dimensionality Reduction
- To optimize computational efficiency and reduce overfitting, we applied dimensionality reduction:
- TruncatedSVD: Similar to PCA, but optimized for sparse data.
- Non-Negative Matrix Factorization (NMF): Ensures all values remain positive, making it suitable for models like Multinomial Naïve Bayes.
- Reduced feature matrix from (22625, 5000) → (22625, 100) while preserving essential information.

# Model Selection and Training
The Logistic Regression model was chosen as the baseline due to its simplicity, efficiency, and effectiveness in text classification tasks. It provided a strong initial benchmark for comparing more complex models.To handle class imbalance, we applied SMOTE, which balanced the distribution of the target variable.

Several other models were explored:
- GridSearchCV: Selected for hyperparameter tuning to optimize parameters.
- Support Vector Machine (SVM) using LinearSVC: Often effective for high-dimensional, sparse data. These alternative approaches allow us to compare their performance with our baseline.
- Multinomial Naive Bayes: A popular model for text classification
  

# Model Evaluation 
After training, the baseline Logistic Regression achieved an accuracy of 83.6%, with strong precision for the "Recommended" class (95%) but moderate performance for the "Not Recommended" class. 

Hyperparameter tuning with GridSearchCV further improved the model, achieving a best cross-validation F1-score of 0.87.

Other models, including Multinomial Naive Bayes and LinearSVC, were explored to compare performance. These models showed varying results, with Naive Bayes reaching 76.1% accuracy and LinearSVC performing similarly to Logistic Regression at 83.5%.

Each model's performance was evaluated based on accuracy, precision, recall, F1-score, and confusion matrix insights.

# Model Interpretability & Feature Importance
In this section, we:
- Examine model coefficients to identify the top positive and negative features (words or n-grams) for each rating class.
- Use interpretability tools like LIME to provide local explanations for individual predictions.
- Prepare visualizations and explanations that are accessible to non-technical stakeholders.
  
Key Takeaways:
- LIME provides interpretability by identifying words that influenced the prediction.
- Negative words (fabric issues, fit problems) led to a "Not Recommended" classification.
- Positive words (love, beautiful, unique) were present but not strong enough to shift the prediction.

# Conclusions
1. Objective Achievement:
We built a predictive model using over 22,000 cleaned reviews.
The model classifies reviews based on ratings, aligning with our goal of understanding customer sentiment and product quality.

2. Data Processing and Feature Engineering:
We performed comprehensive cleaning and preprocessing, including lowercasing, noise removal, tokenization, stopword removal, lemmatization, and n-gram generation.
We transformed the text into numerical features using TF-IDF (incorporating both unigrams and bigrams) and applied optional dimensionality reduction techniques.

3. Model Performance:
Our baseline Logistic Regression model achieved approximately 84% accuracy. The weighted F1-Score was around 0.85, indicating robust performance.
Error analysis and interpretability confirmed key areas for improvement, particularly regarding product fit and fabric quality.

4. Key Insights:
Approximately 60–65% of the reviews are 5-star ratings, suggesting overall high customer satisfaction.
Negative reviews frequently highlight concerns with product fit and fabric quality.
These insights indicate that, while sentiment is largely positive, targeted improvements could further enhance customer experience.

# Recommendations
**Model Recommendations**
- Enhance the Feature Set: Integrate additional structured features such as product category and review date to capture trends over time and differences across product lines.
- Explore Advanced Modeling Techniques: Experiment with transformer-based models (e.g., BERT) or deep learning approaches (e.g., LSTM) to capture subtle language nuances like sarcasm, tone, or ambiguity.
- Consider Data Augmentation: Use data augmentation techniques (e.g., back-translation, synonym replacement) to artificially expand the dataset, especially for underrepresented classes or categories.
- Develop a Continuous Learning Pipeline: Build a production pipeline for continuous model updates as new data arrives.

**Business Recommendations**
- Focus on Fit and Sizing
- Enhance Fabric Quality and Comfort
- Maintain Consistent Product Quality
- Personalize Marketing Strategies

# For more Information:
Review the full analysis on: 
