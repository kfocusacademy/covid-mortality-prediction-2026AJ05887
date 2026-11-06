1.Project Details
----------------
COVID-19 Patient Mortality Prediction

Develop a Machine Learning classification model to predict whether a COVID-19 patient is likely to survive or die based on demographic information, pre-existing medical conditions, and clinical characteristics.

The objective is to develop Machine Learning models that classify COVID-19 patients into two categories:

1 – Died
2 – Survived

2.Machine Learning Models
------------------------
The following five classification algorithms are implemented:

Logistic Regression
Decision Tree
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes

Both baseline and fine-tuned model implementations are included in the source code. 
The fine-tuned models are saved as Joblib files for deployment.

3. Data Preprocessing
-------------------------
 -  Removed INTUBED and ICU columns as these are more specific to post-admission scenario
 -  Converted Date_died column to numeric values of 1(Died) and 2 (survived)
 -  Handling special values in pregnant column 
 -  Train and test split using 80:20 ratio
 -  Applied feature scaling for applicable algorithms 

Features in the dataset
-----------------------------------------------
Category		Columns
Demographics		AGE, SEX, PREGNANT
Medical conditions	DIABETES, COPD, ASTHMA, OBESITY
Other conditions	HIPERTENSION, CARDIOVASCULAR, RENAL_CHRONIC
Clinical information	PNEUMONIA, PATIENT_TYPE, ICU, INTUBED
Other information	USMER, MEDICAL_UNIT, TOBACCO
Target candidate	DATE_DIED


4. Model Evaluations
----------------------
The Models are evaluated using the below specific metrics as per the assignment.
1. Accuracy 
2. AUC Score 
3. Precision 
4. Recall 
5. F1 Score 
6. Matthews Correlation Coefficient (MCC Score)

And additional evaluation outputs:
 - Confusion Matrix
 - Classification Report

5. Project Structure
---------------------

COVID-19-Mortality-Prediction/
│
├── app.py
├── data_model.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── svm.joblib
    ├── knn.joblib
    └── naivebayes.joblib


6. Installation & Execution
-----------------------------
step1: Clone the git repository 
        git clone <YOUR_GITHUB_REPOSITORY_URL>

Step2: cd <project_folder>

step3: Install dependencies
	pip install -r requirements.txt

step4: run streamlit application
	streamlit run app.py

step5: Evaluate the models

	Open the Streamlit application in your browser.
	Upload the test_data.csv file.
	Select an individual classifier or run all five models (Two separate buttons are available)
	View the performance metrics, classification reports and confusion matrices.