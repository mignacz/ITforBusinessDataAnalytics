# ITforBusinessDataAnalytics

This repository contains coursework from my IT for Business Data Analytics Master's degree.

## Contents

### Assignments
1. **Assignment 1**: Deep Learning
   - Notebook: [DeepLearning.ipynb](assignments/DeepLearning.ipynb)
   - Description: The project involves classifying companies from the Forbes Global dataset into two categories based on their market value. The goal is to preprocess the data, create binary labels, train a deep learning model, optimize its performance, and evaluate the results.
   - Programming Language: Python
   - Tools: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, TensorFlow, Keras
   - Platform: Jupyter Notebook, Google Colab
   - Skills Developed: Data Preprocessing, Exploratory Data Analysis (EDA), Visualization, Model Building, Model Evaluation, Hyperparameter Tuning, Reflection & Critical Thinking
   - Outcomes:
      - Dropped 'MVClass' and 'Market_Value' to prevent target leakage and avoid trivializing the classification task.
      - Applied basic feature engineering but should have used the Correlation Matrix to refine feature selection and reduce noise.
      - Preprocessed data by standardizing numerical features and applying one-hot encoding to categorical variables, except for 'Company', which was ordinally encoded to prevent a sparse matrix.
      - Used a simple neural network with one hidden layer (352 neurons) and a dropout layer for regularization. The output layer was designed for binary classification.
      - Future improvements could involve experimenting with more complex architectures, adjusting learning rates, or comparing neural networks to other models like Random Forests. Additionally, increasing epochs and using early stopping would help prevent overfitting.

2. **Assignment 2**: Non Neural Machine Learning
   - Notebook: [Non_NeuralMachineLearning.ipynb](assignments/Non_NeuralMachineLearning.ipynb)
   - Description: This project focuses on preparing, exploring, and building machine learning models to classify arm gestures using smartwatch sensor data. The main goals are to preprocess the data, fine-tune the models, and evaluate their performance using both quantitative and qualitative methods.
   - Programming Language: Python
   - Tools: Pandas, NumPy, Matplotlib, Scikit-learn
   - Platform: Jupyter Notebook, Google Colab
   - Skills Developed: Data Preparation, Modeling, Evaluation, Dimensionality Reduction, Visualization
   - Outcomes:
      - Linear Regression: Struggles without PCA due to sensitivity to outliers. PCA improves performance but remains suboptimal.
      - Random Forest: Excels without PCA, handling non-linear relationships and outliers well.PCA reduces its performance, likely due to information loss.
      - Conclusion: Random Forest without PCA outperforms all models, showing superior predictive power and robustness.


3. **Capstone Project**: 
   - Notebook: [CapstoneProjectPub.ipynb](assignments/CapstoneProjectPub.ipynb)
   - Description: The capstone project focuses on predicting click-through rates (CTR) on advertisements using machine learning models. The goal is to analyze how various features influence user interactions with ads. The primary objective is to identify the features most impactful on CTR and optimize ad placements and formats accordingly.
   - Programming Language: Python
   - Tools: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, RandomForestClassifier, GradientBoostingClassifier, Logistic Regression, SVC, TensorFlow, Keras
   - Platform: Jupyter Notebook, Google Colab
   - Skills Developed: Data Preprocessing, Exploratory Data Analysis (EDA), Visualizing trends and distributions, Identifying correlations, Feature Engineering, Dimensionality reduction using PCA and t-SNE, Machine Learning, Deep Learning, Model Evaluation
   - Outcomes:
      - Hypothesis 1: Effect of Ad Type on CTR: Native ads had a higher CTR than display ads, supported by bar plots and model evaluation metrics.
      - Hypothesis 2: Influence of Ad Position on CTR: Ads placed at the top-left of the page had higher CTR compared to top-right positioned ads.
      - Model Performance: The logistic regression model achieved 76% accuracy, but precision and recall were lower than expected. No single feature strongly predicted CTR; multiple features contributed to the model’s performance.
      - Recommendations: Improve feature engineering and explore additional data for better predictions. Experiment with advanced models and fine-tune hyperparameters for better accuracy and recall.
      - Limitations & Future Research: The dataset had missing values, and additional behavioral data could enhance predictions. Future work should include experimenting with more complex models, expanding the feature set, and conducting A/B tests.


---

### How to Use
- Open the notebooks using Google Colab for an interactive experience:
  - [Assignment 1 on Colab](https://colab.research.google.com/github/mignacz/ITforBusinessDataAnalytics/blob/main/assignments/DeepLearning.ipynb)
  - [Assignment 2 on Colab](https://colab.research.google.com/github/mignacz/ITforBusinessDataAnalytics/blob/main/assignments/Non_NeuralMachineLearning.ipynb)
  - [Capstone Project on Colab](https://colab.research.google.com/github/mignacz/ITforBusinessDataAnalytics/blob/main/assignments/CapstoneProjectPub.ipynb)