1.	GIT Hub Repository:
 https://github.com/kfocusacademy/covid-mortality-prediction-2026AJ05887

2.	Live Streamlit App Link
 https://covid-mortality-prediction-2026aj05887-enyihtxtml9sentxjawega.streamlit.app/

3.Project Details
----------------
COVID-19 Patient Mortality Prediction

Develop a Machine Learning classification model to predict whether a COVID-19 patient is likely to survive or die based on demographic information, pre-existing medical conditions, and clinical characteristics.

The objective is to develop Machine Learning models that classify COVID-19 patients into two categories:

1 – Died
2 – Survived

2.Machine Learning Models
------------------------
The following five classification algorithms are implemented: using 2000+ Data pointers and saved the model. 

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

	Model			Accuracy	F1 Score	Precision	Recall		AUC		MCC
-----------------------------------------------------------------------------------------------------------------------
0	Logistic_FineTuned	0.885		0.9265131	0.889570	0.966666 	0.934383	0.676528
1	DecisionTree_FineTuned	0.8775		0.9205834	0.895899	0.946666	0.903583	0.658479
2	SVM_FineTuned		0.9025		0.9381933	0.894259	0.986666	0.906516	0.729682
3	KNN_FineTuned		0.8825		0.9245585	0.891640	0.96		0.94155		0.669952
4	NaiveBayes_FineTuned	0.2675		0.0456026	1		0.023333	0.903616	0.077053

Observations:

ML Model Name	    Observation about model performance
-------------------------------------------------------------
SVM	                Best overall classification performance among all other models      
                    (90%   Accuracy and 72% MCC). Even in confusion matrix, only 4 out of 300 predicted incorrect for “Died” category. All other models more than 10. 
KNN	                Strong alternative with higher precision and reported AUC.

Logistic Regression	Strong performance and good interpretability

Decision Tree	    Competitive, but slightly behind compared to other 3 models.

Naive Bayes	        Poor default classification performance  compared to other models

Overall, Winner     SVM performed the best model in terms of both mortality and
for your dataset?   recovered, true positive rate is better.
                    Naïve Basic worst performance model even after hyper tuned parameters. 

And additional evaluation outputs:
 - Confusion Matrix - Available in 
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
        git clone   https://github.com/kfocusacademy/covid-mortality-prediction-2026AJ05887

Step2: cd  covid-mortality-prediction-2026AJ05887

step3: Install dependencies
	pip install -r requirements.txt

step4: run streamlit application
	streamlit run app.py

step5: Evaluate the models

	Open the Streamlit application in your browser.
	Upload the test_data.csv file.
	Select an individual classifier or run all five models (Two separate buttons are available)
	View the performance metrics, classification reports and confusion matrices.