

** Telco Customer Churn Prediction** 

**Statement of Purpose** 

Customer churn is a major concern in the telecommunications industry, where retaining existing customers is far more cost-effective than acquiring new ones. With increasing market competition, telecom companies need reliable ways to identify customers who are likely to leave. This project aims to build a predictive model that helps detect potential churners using machine learning techniques. 

We have selected the Telco Customer Churn dataset from Kaggle, which contains \
customer demographics, service usage patterns, billing details, and churn status. The goal is      to analyze this data, develop predictive models, and automate the process of churn prediction. 

Our approach includes: 

- Data preprocessing and transformation 
- Applying models such as Logistic Regression, Random Forest, and XGBoost 
- Using SMOTE to handle class imbalance 
- Evaluating model performance with accuracy, precision, recall, F1-score, and ROC- AUC 
- Automating predictions via a Python script that generates results in Excel format 

This project demonstrates our understanding of predictive analytics and its business impact. It highlights how data-driven decisions can help telecom companies improve customer retention and reduce losses due to churn. 

**Scope of the Project** 

The scope of this project focuses on developing a comprehensive predictive analytics solution to address customer churn in the telecommunications industry. The project encompasses all critical stages of the data science lifecycle, including data acquisition, exploration, preprocessing, modelling, evaluation, automation, and presentation of results. 

Our primary aim is to build a machine learning model capable of accurately classifying customers as likely to churn or not based on historical behavioural and demographic data. This will empower telecom companies to identify high-risk customers in advance and make data-driven decisions to improve retention strategies. 

The major components within the scope of this project include: 

1. Data Acquisition and Understanding 

   We utilize the publicly available Telco Customer Churn dataset from Kaggle. The dataset includes variables such as gender, tenure, service subscriptions, payment methods, and billing details. Initial steps involve examining the structure of the data, handling missing values, and understanding variable distributions. 

2. Data Preprocessing and Feature Engineering 

   The dataset undergoes rigorous cleaning, including encoding categorical variables, scaling numerical features, and handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique). Additional feature selection and transformation techniques are applied to improve model performance. 

3. Model Development and Evaluation 

   Several classification models are implemented, including Logistic Regression, Random Forest, and XGBoost (Extreme Gradient Boosting). We conduct hyperparameter tuning using GridSearchCV and evaluate models based on metrics like accuracy, precision, recall, F1-score, and AUC-ROC. This comparative approach ensures robustness and reliability in our predictions. 

4. Automation and Application: 

   The final model is serialized into a .pkl file for reuse. A separate Python script is developed to automate the prediction process by accepting a new customer dataset in CSV format and outputting the results including churn predictions and confidence scores—to an Excel file. This makes the solution deployable in a business environment. 

5. Analytical Integration: 

   To align with course learning outcomes, we incorporate univariate and bivariate analysis, transformations (log, non-linear), and regression insights to strengthen our understanding of underlying patterns and variable relationships. 

6. Deliverables: 

   The final submission includes the original dataset (telco\_customer\_churn.csv), Google Colab[ (Assignment_2_Group_9.ipynb) ](https://colab.research.google.com/drive/15QWBUuQc5i6Dahx8-RhwDI_SDzan36fj?usp=sharing)for model training, Python script for prediction (predict\_using\_pkl\_file.py), Excel output file (churn\_predictions\_output.xlsx), project report (Assignment\_2\_Group\_9.pdf), presentation slides, and a recorded video walkthrough of the project. 

By defining clear boundaries and deliverables, this project ensures that all components from technical development to business applicability are covered. The scope is intentionally broad to reflect real-world analytics workflows yet focused enough to demonstrate tangible impact through actionable customer churn insights. 

**Background Research and Literature** 

Customer churn prediction has become increasingly important in the telecommunications sector, where retaining customers is more cost-effective than acquiring new ones. Predictive analytics helps companies identify customers likely to leave and take timely action to reduce churn. Various studies have demonstrated the effectiveness of machine learning techniques in addressing this issue. 

Idris et al. (2012) introduced an ensemble approach using RotBoost combined with feature selection techniques, which significantly improved the prediction accuracy in telecom churn datasets. Their work emphasized the value of combining multiple classifiers and selecting the most informative features to build reliable predictive models. 

Ahmad et al. (2019) focused on customer churn prediction using machine learning in a big data environment. They evaluated models such as decision trees, random forests, and gradient boosting, showing how early identification of high-risk customers can lead to improved marketing strategies and customer satisfaction. 

These research findings validate the approach adopted in our project. We apply Random Forest and XGBoost models, use feature engineering, and handle data imbalance using SMOTE. Our methodology is informed by academic best practices and is designed to deliver a solution that is both accurate and actionable for business use. 

**Design and Data Collection Methods** 

The dataset used in this project, titled “Telco Customer Churn,” was sourced from Kaggle, a well-known platform that provides curated and structured datasets for data analysis and machine learning tasks. The dataset is provided by IBM and includes real-world customer data from a telecom company. It contains information about 7,043 individual customers and 21 features, covering demographics, account details, service usage, and churn status. No manual data collection was required, as the dataset is publicly available and provided in CSV format. 

Each row represents a customer; each column contains customer’s attributes described on the column Metadata. 

The data set includes information about: 

- Customers who left within the last month – the column is called Churn 
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies 
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges 
- Demographic info about customers – gender, age range, and if they have partners and dependents 

After importing the dataset into a Google Colab, an initial data exploration phase was conducted to understand the feature types, distributions, and potential quality issues.  Summary statistics, visual inspection, and value counts were used to identify: 

- Data types of each column (numeric, categorical, boolean). 
- Target variable distribution (Churn) indicating class imbalance. 
- Presence of missing or improperly formatted data, especially in TotalCharges. 

The dataset contained a few rows with missing or incorrectly typed values: 

- Missing Values: Rows with empty values in the TotalCharges column were identified and dropped since they represented a negligible portion of the dataset. 
- Data Type Correction: TotalCharges was originally read as a string. This was converted to a numeric format to allow proper analysis. 
- Consistency Checks: Columns like SeniorCitizen, which was numeric (0 or 1), were converted to categorical format for better interpretability. 

The dataset had a mix of binary and multi-class categorical variables which required encoding: 

- Label Encoding was applied to binary variables such as Partner, Dependents, and PaperlessBilling (Yes/No to 1/0). 
- One-Hot Encoding was used for variables like InternetService, Contract, and PaymentMethod, which have multiple non-ordinal categories. 
- Standardization was performed on continuous numeric features such as MonthlyCharges and TotalCharges using StandardScaler to ensure that features were on the same scale. 

After cleaning and encoding, the dataset was split into features (X) and target (y). The final pre-processed dataset was now suitable for training classification models. At this stage, the data was ready for the next step modelling and evaluation which is discussed in the Methodology section. 

This structured and thorough approach to data design and preprocessing ensured that the dataset was clean, consistent, and suitable for robust predictive analytics modelling. 

**Methodology/Strategies** 

This project follows a structured machine learning pipeline to develop a robust churn prediction model. The methodology consists of multiple stages from exploratory data analysis to final model deployment with a focus on ensuring statistical validity, automation, and business relevance. 

1. Exploratory Data Analysis (EDA) 

   Before modelling, we performed statistical exploration to understand data characteristics and identify relationships: 

- Univariate Analysis: Summary statistics (mean, median, standard deviation) and plots (histograms, boxplots) were used to examine distributions of individual variables. 
- Bivariate Analysis: Cross-tabulations and bar plots were used to assess relationships between independent features (e.g., tenure, contract type) and the target variable (Churn). 
- Correlation Matrix: Pearson correlation was used for continuous variables to detect linear relationships and multicollinearity. 

![](Aspose.Words.fa035bd4-1b42-4abd-ac53-d21b02035e40.002.jpeg)

2. Data Transformation 

   To meet model assumptions and improve performance: 

- Log-Transformation: Applied to skewed numeric features (e.g., TotalCharges) to normalize distributions. 
- Encoding: Label encoding and one-hot encoding were used to convert categorical data into machine-readable format. 
- Feature Scaling: Standardization was performed on numerical variables to ensure equal weightage in distance-based algorithms. 
3. Handling Class Imbalance 

   Since the Churn variable was imbalanced (~26% churned vs. 74% not churned), we used SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes during model training. This avoids model bias toward the majority class and improves recall and F1-score. 

![](Aspose.Words.fa035bd4-1b42-4abd-ac53-d21b02035e40.003.png)

4. Model Development 

   We experimented with multiple classification algorithms to identify the best- performing model: 

- Logistic Regression: Used as a baseline model for its simplicity and interpretability. 
- Random Forest Classifier: Employed to capture non-linear relationships and reduce overfitting via ensemble averaging. 
- XGBoost Classifier: Chosen for its superior accuracy, speed, and ability to handle feature interactions and missing data internally. 

Each model was trained on an 80:20 train-test split, and hyperparameters were optimized using GridSearchCV for the best combination of accuracy and generalization. 

5. Evaluation Metrics 

   ![](Aspose.Words.fa035bd4-1b42-4abd-ac53-d21b02035e40.004.jpeg)

   To assess the effectiveness of each model, we used: 

- Accuracy: Overall correctness of predictions. 
- Precision & Recall: To evaluate false positive/negative rates. 
- F1-Score: Harmonic mean of precision and recall. 
- AUC-ROC Curve: To evaluate classification thresholds and distinguish between churned and retained customers. 

Cross-validation was applied to ensure that results were not biased due to random sampling. 

![](Aspose.Words.fa035bd4-1b42-4abd-ac53-d21b02035e40.005.jpeg)

![](Aspose.Words.fa035bd4-1b42-4abd-ac53-d21b02035e40.006.jpeg)

6. Model Deployment Strategy 

   The final model (XGBoost) was saved using joblib into a .pkl file. An automation script (predict\_using\_pkl\_file.py) was developed to: 

- Accept new customer data via CSV 
- Preprocess the data using the same steps as training 
- Predict churn probability and class 
- Output results to churn\_predictions.xlsx with prediction scores 

This automation supports business use-cases where batch or real-time predictions are needed with minimal manual intervention. 

**Business Impact** 

Customer churn is a significant challenge in the telecommunications industry, directly affecting profitability and market competitiveness. Acquiring a new customer is often five times more expensive than retaining an existing one. Thus, implementing a predictive churn model allows telecom companies to proactively identify customers at risk of leaving and intervene with targeted retention strategies. 

By analyzing historical customer data and building a machine learning model, this project empowers businesses to: 

- Reduce Churn Rate: Early identification of high-risk customers enables companies to offer personalized promotions, service upgrades, or improved customer support. 
- Optimize Marketing Budget: Rather than spending uniformly on retention efforts, companies can focus resources on those most likely to churn. 
- Enhance Customer Experience: By understanding churn drivers such as long tenure without upgrades or unsatisfactory contract types, telecom providers can make informed product and service adjustments. 
- Increase Revenue: Retaining even a small percentage of customers can lead to significant long-term revenue gains. 

The deployment-ready Python script further enhances business usability by allowing seamless integration into internal CRM or BI tools, enabling real-time churn predictions with minimal technical intervention. 

**Conclusion** 

This project successfully developed a predictive analytics solution for customer churn using publicly available data and machine learning techniques. By following a systematic data science pipeline spanning data acquisition, preprocessing, model training, and deployment the project demonstrated how predictive models can provide actionable business insights. 

Among the models tested, XGBoost performed best in terms of accuracy and robustness, and the final output was automated for practical use. The insights derived from the data analysis and the power of machine learning enable companies to move from reactive churn management to proactive customer retention. 

Future improvements could include integrating customer feedback data, service usage logs, or real-time behavioural tracking to further improve prediction accuracy and business relevance. 

**References** 

Ahmad, A., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using 

machine learning in big data platform. Journal of Big Data, 6(1), 1–24. 

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic 

Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.[ https://doi.org/10.1613/jair.953 ](https://doi.org/10.1613/jair.953)

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of 

the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.[ https://doi.org/10.1145/2939672.2939785 ](https://doi.org/10.1145/2939672.2939785)

IBM Analytics. (n.d.). CRISP-DM 1.0: Step-by-step data mining guide. Retrieved from IBM 

Data Science Community:[ https://www.ibm.com/docs/en/spss- modeler/saas?topic=guide-crisp-dm-10-step-step-data-mining ](https://www.ibm.com/docs/en/spss-modeler/saas?topic=guide-crisp-dm-10-step-step-data-mining)

IBM Sample Data. (n.d.). Telco Customer Churn Dataset. Retrieved from Kaggle: 

[https://www.kaggle.com/datasets/blastchar/telco-customer-churn ](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Idris, A., Khan, A., & Lee, Y. S. (2012). Intelligent churn prediction in telecom: employing 

mRMR feature selection and RotBoost based ensemble classification. Applied Intelligence, 39(3), 659–672. 

Van Veen, F. (2017). The Data Science Process. Retrieved from Towards Data Science: 

[https://towardsdatascience.com/the-data-science-process-a-definition-610701201c3e ](https://towardsdatascience.com/the-data-science-process-a-definition-610701201c3e)
15 
