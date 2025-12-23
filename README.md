Audit Risk using Logistic Regression and Random Forest


1. Problem
The primary problem addressed in this project is the identification of fraudulent firms based on audit risk data. Auditors face the challenge of examining vast amounts of financial and operational data to determine if a firm poses a significant risk. The goal is to build a predictive model that can classify a firm as having "Risk" (1) or "No Risk" (0) based on various quantitative and qualitative features. Thus, this is a binary classification problem.


2. Motivation
Traditional auditing processes are labor-intensive, time-consuming, and subject to human error. With the increasing volume of data, manual detection of fraudulent patterns has become inefficient. Thus, machine learning algorithms were applied to this domain to:
Automate Risk Detection: Create a system that can instantly flag high-risk firms.
Improve Accuracy: Leverage historical data patterns to make more consistent decisions than manual review.
Resource Allocation: Allow auditors to focus their limited time and resources on firms flagged as high-risk by the model, rather than randomly sampling.


3. Dataset Information
The dataset used for this project is from audit_risk.csv.
Original Dimensions: 776 rows and 27 columns.
Target Variable: Risk (Binary: 0 or 1).
Features: This dataset includes the following audit metrics: Sector_score, LOCATION_ID, PARA_A, Score_A, Risk_A, PARA_B, Score_B, Risk_B, TOTAL, numbers, Score_B (this column is different from the initial ‘Score_B’), Risk_C, Money_Value, Score_MV, Risk_D, District_Loss, PROB, RiSk_E, History, Prob, Risk_F,  Score, Inherent_Risk, CONTROL_RISK, Detection_Risk, and Audit_Risk.
Data Types: Most columns are numeric (float/int), except for LOCATION_ID, which contains object data (categorical).


4. Feature Processing and Feature Engineering
To prepare the data for machine learning, several preprocessing steps were undertaken.

4.1. Surface Cleaning:
Duplicate Column Investigation: Columns like Score_B vs. Score_B (auto-renamed to Score_B.1 within Jupyter Notebook) and PROB vs. Prob appeared redundant. However, value counts showed they were not identical. Therefore, they were retained as independent features but renamed (to Score_B1, Score_B2, PROB_1, PROB_2) to maintain data integrity while improving clarity.
Typo Correction: Column RiSk_E was renamed to Risk_E for consistency.
Zero Variance Removal: The Detection_Risk column contained only a single unique value (0.5) across all rows. Because it offered zero variance for the model, it was dropped.
Missing Values: There was one missing value in the dataset, which appeared in one of the rows of the Money_Value column. Since this represented a negligible portion of the data (1/776), the row was dropped rather than imputed, as imputation could introduce bias for such a small loss of data.

4.2. Train-Test Split
Method: StratifiedShuffleSplit (70% Train, 30% Test).
Rationale: This ensures that the proportion of fraudulent to non-fraudulent firms remains consistent between the training and testing sets, preventing the model from learning from a skewed distribution.

4.3. Feature Scaling
Method: MinMaxScaler.
Rationale: Numeric features (excluding the categorical LOCATION_ID) were scaled to a range of [0, 1]. This is crucial for algorithms like Logistic Regression, which are sensitive to the feature coefficients. Without scaling, features with larger raw values (like those in TOTAL and Inherent_Risk) would disproportionately influence the model weights compared to smaller features (like Score_A, Score_B, and Score_C).

4.4. Encoding Categorical Data
Method: OneHotEncoder on LOCATION_ID.
Rationale: LOCATION_ID is a nominal categorical variable. Using label encoding (assigning numbers 1, 2, 3...) would imply an ordinal relationship (i.e., location 4 is "greater" than location 2) which does not exist. One-hot encoding creates binary columns for each location, allowing the model to treat each location independently without falsely assuming ordinal relationships.


5. Machine Learning Model Development
Two primary algorithm families were applied to the dataset: Linear Models and Ensemble Models.

5.1. Logistic Regression (Linear Model)
Baseline Model: A standard Logistic Regression model was fitted with max_iter=1000 to ensure convergence.
Regularized Models (L1 & L2): LogisticRegressionCV was used with max_iter=5000 (to ensure convergence) and both Lasso (L1) and Ridge (L2) penalties using 4-fold cross-validation.
Rationale: L1 regularization helps in feature selection by driving some feature weights to zero, while L2 regularization shrinks the contributions of the features. Cross-validation ensures the regularization strength (C) is optimized for generalization.

5.2. Random Forest Classifier (Ensemble Model)
Configuration: A Random Forest was initialized with oob_error=True (Out-of-Bag error provided) and warm_start=True (so that subsequent .fit() calls will build on top of pre-existing trees rather than re-building everything from scratch).
Hyperparameter Tuning: The model was iteratively trained with an increasing number of trees (estimators) ranging from 15 to 400.
Rationale: Random Forest handles non-linear relationships better than Logistic Regression and is generally robust against overfitting (compared to standard decision trees via DecisionTreeClassifier). The OOB error was monitored to find the optimal number of trees where the error stabilized, avoiding unnecessary computational cost.


6. Evaluating the Result/Metrics
The evaluation utilized multiple metrics to ensure a holistic view of performance.

6.1. Quantitative Metrics
Accuracy: High accuracy indicates predictions and test results align extremely well.
Precision: High precision indicates that when the model predicts Risk, it is usually correct.
Recall: High recall indicates the model successfully caught most of the actual fraudulent instances.
F1-Score: It is the harmonic mean of Precision and Recall. A high F1-Score was observed, indicating models can make accurate predictions that minimize both false positives and false negatives.
ROC-AUC: All models achieved an Area Under the Curve (AUC) > 0.99, indicating the model separates classes almost perfectly.

6.2. Visualizations
ROC Curves: The ROC curves for all models hugged the top-left corner. For all of the Logistic Regression instances, the area-under-the-curve (AUC) for the ROC curves were just below 1. This was not the case for the ROC curves in the Random Forest instances. For both cases where Audit_Risk was kept and then dropped (see NOTE as a sub-point under this), AUC = 1.0, indicating perfect Random Forest models.
NOTE: Upon further analyses, Audit_Risk was dropped after seeing that there was some value within Audit_Risk that allowed this feature to perfectly separate the ‘Risk’ class. Even after dropping Audit_Risk, AUC remained at 1.0. Thus, this dataset should be analyzed in greater detail.
Precision-Recall Curves: The Precision-Recall Curve (PRC) shows how well a model balances precision (how many predicted positives are correct) and recall (how many actual positives are detected).
Confusion Matrices: The confusion matrices showed minimal misclassifications. For all cases of Logistic Regression, there were approximately 6-8 misclassifications out of 233 test samples. For all cases of Random Forest, there were approximately 0-1 misclassifications out of 233 test samples.
OOB-Error-to-Trees Curves (Random Forest): Out-of-bag (OOB) error is an estimate of the error associated with each random forest. For the initial case of Random Forest where the Audit_Risk column was kept, the out-of-bag (OOB) error remained relatively stable after 30 trees. However, for the case of Random Forest where the Audit_Risk column was dropped, the out-of-bag error was at its low at 100 trees, but climbed as the number of trees surpassed 100.
Feature Importance (Random Forest): An important feature is one that is a significant predictor of Risk. For the initial case of Random Forest where the Audit_Risk column was kept, Audit_Risk, Inherent_Risk, Score, and TOTAL were the most important features (in that order). However, after the Audit_Risk column was dropped, Score, TOTAL, and Risk_D became the top three most important features. The order of the subsequent features were different compared to the order in the case where Audit_Risk was kept.

8. Conclusion
This project successfully demonstrated the viability of using machine learning to automate audit risk classification. Both Logistic Regression (L1/L2) and Random Forest models achieved high performance, with accuracies well above 96%.

While both models performed extremely well, the Random Forest classifier is the preferred model for deployment. Its inherent robustness against overfitting and ability to provide interpretable feature importance rankings make it the superior option within a business context, where explaining "why" a firm was flagged is necessary.
A critical phase of this project involved identifying and attempting to resolve overfitting with the Random Forest approach. The inclusion of Audit_Risk resulted in an accuracy of 100% for the initial Random Forest model, which is unrealistic. Removing this feature caused a slight expected drop in accuracy but increased the validity of the model.
After the removal of Audit_Risk, the Random Forest model shifted reliance from one dominant feature to a few important features. However, despite the high testing accuracy, skepticism remains regarding potential overfitting for the model. The high correlation between several remaining features suggests that the model might still be learning from "proxies" of the target variable rather than independent risk factors. Future work should involve rigorous testing on completely external datasets to ensure the model isn't just memorizing specific patterns of the dataset from audit_data.csv.
For an auditing firm, this tool can become part of a powerful triage system. Rather than replacing auditors, it will optimize for resource allocation. Instead of random sampling, the firm can run this model on thousands of clients instantly, identifying the top x% highest-risk cases. Because time and effort are limited, this will allow expert auditors to quickly pin down firms that are statistically most likely to be fraudulent, thereby increasing detection rates and operational efficiency.

