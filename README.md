### Project: Persian News Classification

#### Dataset Overview:
The dataset contains news articles in Persian with the following columns:

1. newsid: Unique identifier for each news article.
2. title: The title of the news article.
3. body: The content/body of the news article.
4. date: The date when the news article was published.
5. time: The time when the news article was published.
6. category: The category to which the news article belongs (e.g., Political, Social, Economic, etc.).

---

### Project Steps:

1. Exploratory Data Analysis (EDA):
   - I started by performing EDA to analyze the dataset and check for any missing values. I found that there were no missing values in the dataset, which made it ready for further processing.

2. Loading Stop Words:
   - I uploaded a list of stop words (common words like "the", "is", etc., that do not carry significant meaning in text analysis) to remove them later in the preprocessing steps.

3. Text Preprocessing using NLTK and Hazm:
   - I utilized NLTK and Hazm libraries, which are specialized in Persian text processing. Using these libraries, I performed the following steps:
     - Created a new dataframe with title, body, and category columns.
     - Tokenized the text (split the text into individual words) using a line-by-line process.
     - Removed stop words from the tokenized text.
     - Applied stemming (using the stemmer.stem() function) to reduce words to their root forms.

4. Feature Extraction with TF-IDF:
   - I used TF-IDF Vectorizer to transform the text data into numerical features, which is useful for machine learning algorithms to understand the text.

5. Label Encoding:
   - I encoded the category column using LabelEncoder. There were 10 unique categories in the dataset (e.g., Political, Social, Economic, etc.), so I used the LabelEncoder to assign a numerical label to each category.

6. Model Building:
   - I used the following algorithms to classify the news articles:

   - SVC (Support Vector Classifier):
     - I applied an SVC model with a linear kernel.
     - The model achieved 85% accuracy on the test data.
     - I visualized the confusion matrix, which showed how well the model predicted each category.

   - CatBoost:
     - I used the CatBoost algorithm with the following parameters:
       - Learning Rate: 0.05
       - Number of Iterations: 300
       - Task Type: GPU (to speed up training).
     - The model achieved 80% accuracy on the test data.

   - Random Forest:
     - I also tried the Random Forest algorithm, which achieved 75% accuracy.

7. Model Saving:
   - The best-performing model was SVC with 85% accuracy. I saved the following components for future use:
     - The trained SVC model.
     - The Tokenizer.
     - The Stemmer.
     - The TF-IDF Vectorizer.
     - The Label Encoder.

---

### Conclusion:

This project demonstrates how text classification techniques can be applied to Persian news articles for automatic categorization. The SVC model performed the best with 85% accuracy, followed by CatBoost (80%) and Random Forest (75%). Key techniques such as TF-IDF, stopword removal, stemming, and Label Encoding played an essential role in preprocessing and feature extraction. The project also showed the practical use of different machine learning algorithms and the importance of selecting the right model for a classification task.

---

### Skills Demonstrated:
1. Text Preprocessing: Handling Persian text using Hazm and NLTK, including tokenization, stopword removal, and stemming.
2. Feature Extraction: Using TF-IDF Vectorizer to convert text data into numerical features for machine learning.

3. Model Building: Implementing and training machine learning algorithms such as SVC, CatBoost, and Random Forest for classification tasks.
4. Hyperparameter Tuning: Using CatBoost with custom parameters like learning rate and GPU for faster training.
5. Model Evaluation: Evaluating models using accuracy and confusion matrix to assess performance.
6. Model Deployment: Saving and storing the best model along with necessary preprocessing components like tokenizer, stemmer, vectorizer, and label encoder for future use.

This project showcases an understanding of text classification and natural language processing (NLP) techniques, which can be applied to a wide range of text-based tasks, including news categorization.