# Fake Job Posting Classification using NLP

## Overview
This Jupyter Notebook (`Fake_Job_Cassification_with_NLP.ipynb`) implements a comprehensive machine learning and deep learning pipeline to classify job postings as legitimate or fraudulent using Natural Language Processing (NLP) techniques. The project addresses the growing issue of fake job scams by analyzing textual content from job listings to identify patterns indicative of fraud.

The notebook covers the entire workflow: data loading and merging, preprocessing, exploratory data analysis (EDA), feature extraction, model training, and evaluation. It evaluates multiple models including traditional machine learning algorithms and deep learning architectures.

## Dataset
The notebook uses two datasets:
- **Original Dataset** (`fake_job_postings.csv`): Contains 17,880 real-world job postings with features like title, location, company_profile, description, requirements, benefits, etc. Only a small fraction are labeled as fraudulent.
- **Synthetic Dataset** (`synthetic_dataset.csv`): 5,000 additional postings generated using Gretel.ai to augment fraudulent examples and balance the dataset.

The datasets are merged into a single DataFrame of 22,880 postings. Key columns include textual fields and the binary target `fraudulent` (0 for real, 1 for fake).

## Methodology

### 1. Data Loading and Initial Exploration
- Import necessary libraries (pandas, numpy, nltk, sklearn, tensorflow, etc.).
- Load and inspect datasets using `pd.read_csv()`.
- Check shapes, data types, and missing values with `df.info()` and `df.isnull().sum()`.

### 2. Data Cleaning and Preprocessing
- **Handling Missing Values**: Replace nulls in categorical columns with 'Unspecified'; use np.nan for numerical columns.
- **Feature Engineering**:
  - Map `required_education` to standardized categories (e.g., "Bachelor's Degree" → "Bachelor").
  - Extract country codes from `location`.
  - Group industries into broader categories (e.g., "Technology" for software-related fields).
  - Convert binary columns (e.g., `has_company_logo`) to descriptive text.
- **Text Combination**: Merge all textual columns into a single `full_text` column for analysis.
- **NLP Preprocessing**:
  - Tokenization, lowercasing, stopword removal, lemmatization.
  - Create cleaned text and token lists.

### 3. Exploratory Data Analysis (EDA)
- **N-gram Analysis**: Extract and visualize top 20 bigrams/trigrams for real vs. fake postings (before/after cleaning) using `nltk.ngrams` and `Counter`.
- **Word Clouds**: Generate word clouds for cleaned text using `WordCloud` to highlight frequent words.
- **Text Length Distributions**: Plot histograms for lengths of key text columns using `seaborn.histplot`.
- **Industry and Education Analysis**: Bar plots of top industries for real/fake postings.

### 4. Feature Extraction (Vectorization)
- **TF-IDF**: Use `TfidfVectorizer` with min_df=0.01, max_df=0.99 for training/test matrices.
- **Count Vectorization**: Use `CountVectorizer` similarly.
- **Word2Vec**: Train a `gensim.models.Word2Vec` model on tokenized text; generate embeddings by averaging word vectors.
- **Dimensionality Reduction**: Apply `TruncatedSVD` (PCA) to reduce TF-IDF/Count matrices to 230 dimensions.

### 5. Model Training and Evaluation
- **Train-Test Split**: 60-40 split using `train_test_split` with random_state=46.
- **Models Evaluated**:
  - **Naive Bayes**: `MultinomialNB(alpha=0.6)` on TF-IDF/Count.
  - **Random Forest**: `RandomForestClassifier` with tuned hyperparameters on reduced TF-IDF/Word2Vec.
  - **SVM**: `SVC(kernel='linear', C=0.5)` on reduced TF-IDF/Word2Vec.
  - **RNN/LSTM**: Keras Sequential models with Bidirectional LSTMs, Dropout; trained on reshaped TF-IDF or Word2Vec embeddings.
- **Evaluation Metrics**: Classification reports (`precision`, `recall`, `f1-score`) and confusion matrices using `sklearn.metrics`.

## Results Summary
The models were evaluated on accuracy, precision, recall, and F1-score for both classes (0: real, 1: fraudulent). Below are key results from the notebook's evaluations:

### TF-IDF Vectorization
- **Naive Bayes**:
  - Accuracy: 97%
  - Precision (Fraudulent): 0.93, Recall: 0.93, F1: 0.93
  - Overall strong performance on text frequency features.
- **Random Forest** (after SVD reduction):
  - Accuracy: 93%
  - Precision (Fraudulent): 0.83, Recall: 0.83, F1: 0.83
- **SVM** (after SVD reduction):
  - Accuracy: 98%
  - Precision (Fraudulent): 0.96, Recall: 0.96, F1: 0.96
- **RNN/LSTM** (reshaped TF-IDF):
  - Accuracy: 96-97%
  - Precision (Fraudulent): 0.92-0.93, Recall: 0.92-0.93, F1: 0.92-0.93

### Word2Vec Embeddings
- **Random Forest**:
  - Accuracy: 94%
  - Precision (Fraudulent): 0.87, Recall: 0.87, F1: 0.87
- **SVM**:
  - Accuracy: 97%
  - Precision (Fraudulent): 0.93, Recall: 0.93, F1: 0.93
- **RNN/LSTM** (with Word2Vec embeddings):
  - Accuracy: 99%
  - Precision (Fraudulent): 0.98, Recall: 0.98, F1: 0.98
  - Highest performance, with LSTM Test Accuracy: 99% (from evaluation).

Deep learning models (Bidirectional LSTM) outperform traditional ML, especially with Word2Vec embeddings capturing semantic context. Confusion matrices showed low false positives/negatives for top models.

## Prerequisites
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, regex, nltk, wordcloud, scikit-learn, tensorflow/keras, plotly, gensim, transformers, torch
- NLTK data: punkt, stopwords, wordnet, omw-1.4 (download via `nltk.download()`)

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn regex nltk wordcloud scikit-learn tensorflow keras plotly gensim transformers torch
   ```
3. Download NLTK data in Python:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Running the Notebook
1. Place `fake_job_postings.csv` and `synthetic_dataset.csv` in an accessible directory (update paths in the notebook if needed).
2. Open the notebook in Jupyter Notebook/Lab.
3. Run cells sequentially:
   - Start with imports and data loading.
   - Proceed through preprocessing, EDA, vectorization, and modeling.
   - Models are trained and evaluated in separate sections.
4. Note: Deep learning sections may require GPU for faster training; adjust epochs/batch_size as needed.

## Key Code Snippets
- **Text Preprocessing Function**:
  ```python
  def description_lemmatize(revised_tokens):
      wnl = nltk.WordNetLemmatizer()
      return [wnl.lemmatize(tokens) for tokens in revised_tokens]
  ```
- **Model Creation (LSTM)**:
  ```python
  def create_lstm_model(input_shape):
      model = Sequential()
      model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
      model.add(Bidirectional(LSTM(64)))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      return model
  ```

## Conclusion
This notebook demonstrates effective use of NLP for fraud detection in job postings. Bidirectional LSTMs with Word2Vec achieve near-perfect accuracy, highlighting the value of semantic embeddings. Future enhancements could include BERT integration or deployment as a web tool.


