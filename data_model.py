# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB  # model = GaussianNB()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import os
import joblib

# Global Variables
results = []
var_accuracy = 0.00
var_f1_score = 0.00
var_precision = 0.00
var_recall = 0.00
var_auc = 0.00
var_mcc = 0.00

# %%
# Data cleansing 
X = pd.DataFrame()
y = pd.DataFrame()
cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

scoring = {
        'MCC': 'matthews_corrcoef',
        'F1': 'f1_macro',
        'AUC': 'roc_auc'
    }
classification_reports = []
confusion_matrices = []

# %%
def load_data_set(dataset_file):
    df=pd.read_csv(dataset_file)
    return df

# df = load_data_set('CovidData_Small.csv')

def data_cleansing(df,dataset_name):
    if dataset_name == "Covid":
        df = df.copy()
        # Drop columns intubed and ICU as the model gets gets information about interventions that may happen after admission. 
        df.drop(columns=['INTUBED','ICU'],inplace=True)
        # Replacing date_died column to keep only 1 - Died, 2 Survived for better performance 
        df['DATE_DIED'] = [2 if row =='9999-99-99' else 1 for row in df['DATE_DIED']]
        # Fix codes for Pregnant 
        df['PREGNANT'] = df['PREGNANT'].replace(97,2)
        df['PREGNANT'] = df['PREGNANT'].replace(98,2)
        y = df['DATE_DIED']
        X = df.drop('DATE_DIED', axis=1)
        return X,y
    else:
        raise ValueError("Dataset not supported")

def prepare_data(dataset_file):

    # Load dataset
    df = load_data_set(dataset_file)

    # Clean dataset
    X, y = data_cleansing(df, "Covid")

    # Split dataset
    train_x, test_x, train_y, test_y = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Save Test Data to a seperate csv file
    test_data = test_x.copy()
    test_data['DATE_DIED'] = test_y
    test_data.to_csv("covid_test_data.csv", index=False)

    
    # Scale features
    scaler = StandardScaler()

    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    return (
        train_x, test_x, train_y, test_y,
        train_x_scaled, test_x_scaled, scaler
    )

# Logistic Regression model
def logistic_reg(train_x_scaled,train_y,test_x_scaled,test_y):
    print("Start of the logistic model")
    log_reg = LogisticRegression(max_iter=5000,
                                    solver='lbfgs',
                                    random_state=42)
    # model training 
    log_reg.fit(train_x_scaled, train_y)
    
    # get predicted values 
    y_pred = log_reg.predict(test_x_scaled)
    y_prob = log_reg.predict_proba(test_x_scaled) 
    # Evaluating metrics
    # Identify the probability column for death (class 1)
    death_index = list(log_reg.classes_).index(1)
    evaluate_metrics("Logistic",test_y,y_pred,y_prob,death_index)
    
    # returning model
    return log_reg

# %%
def SVM(train_x_scaled,train_y,test_x_scaled,test_y):
    print("Start of the SVM")
    svm_model = SVC(kernel='rbf',probability=True,random_state=42)
    svm_model.fit(train_x_scaled, train_y)
    # get predicted values 
    y_pred = svm_model.predict(test_x_scaled)
    y_prob = svm_model.predict_proba(test_x_scaled) 
    # Evaluating metrics
    # Identify the probability column for death (class 1)
    death_index = list(svm_model.classes_).index(1)
    evaluate_metrics("SVM",test_y,y_pred,y_prob,death_index)
    return svm_model

# %%
def decision_tree(train_x,train_y,test_x,test_y):
    print("Start of the decision tree model")
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_x, train_y)
    # get predicted values 
    y_pred = decision_tree.predict(test_x)
    y_prob = decision_tree.predict_proba(test_x) 
    # Evaluating metrics
    # Identify the probability column for death (class 1)
    death_index = list(decision_tree.classes_).index(1)
    evaluate_metrics("DecisionTree",test_y,y_pred,y_prob,death_index)
    return decision_tree
    
# %%
def KNN(train_x_scaled,train_y,test_x_scaled,test_y):
    print("Start of the K nearest neighbors")
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(train_x_scaled, train_y)
    # get predicted values 
    y_pred = model_knn.predict(test_x_scaled)
    y_prob = model_knn.predict_proba(test_x_scaled) 
    # Evaluating metrics
    # Identify the probability column for death (class 1)
    death_index = list(model_knn.classes_).index(1)
    evaluate_metrics("KNN",test_y,y_pred,y_prob,death_index)
    return model_knn
    
# %%
def NaiveBayes(train_x,train_y,test_x,test_y):
    print("Start of the Naive Bayes")
    nb_model = GaussianNB()
    nb_model.fit(train_x, train_y)
    # get predicted values 
    y_pred = nb_model.predict(test_x)
    y_prob = nb_model.predict_proba(test_x) 
    # Evaluating metrics
    # Identify the probability column for death (class 1)
    death_index = list(nb_model.classes_).index(1)
    evaluate_metrics("NaiveBayes",test_y,y_pred,y_prob,death_index)
    return nb_model

# %%

def Logistic_ft(train_x,train_y,test_x,test_y):

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000))
    ])

    lr_params = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__class_weight': [None, 'balanced']
    }

    lr_grid = GridSearchCV(
        lr_pipeline,
        lr_params,
        cv=cv,
        scoring=scoring,
        refit='MCC',
        n_jobs=-1
    )

    lr_grid.fit(train_x, train_y)
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        matthews_corrcoef
    )

    # Best model selected by GridSearchCV
    lr_model = lr_grid.best_estimator_

    # Predictions
    y_pred = lr_model.predict(test_x)
    y_prob = lr_model.predict_proba(test_x) 
    # Identify the probability column for death (class 1)
    death_index = list(lr_model.classes_).index(1)
    evaluate_metrics("Logistic_FineTuned",test_y,y_pred,y_prob,death_index)
    return lr_model

# %%
def DecisionTree_ft(train_x,train_y,test_x,test_y):

    dt_model = DecisionTreeClassifier(random_state=42)

    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'class_weight': [None, 'balanced']
    }

    dt_grid = GridSearchCV(
        dt_model,
        dt_params,
        cv=cv,
        scoring=scoring,
        refit='MCC',
        n_jobs=-1
    )

    dt_grid.fit(train_x, train_y)
    # Best model selected by GridSearchCV
    best_model = dt_grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(test_x)
    y_prob = best_model.predict_proba(test_x)

    # Identify the probability column for death (class 1)
    death_index = list(best_model.classes_).index(1)
    evaluate_metrics("DecisionTree_FineTuned",test_y,y_pred,y_prob,death_index)
    return best_model
# %%

def SVM_ft(train_x,train_y,test_x,test_y):
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True, random_state=42))
    ])

    svm_params = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }

    svm_grid = GridSearchCV(
        svm_pipeline,
        svm_params,
        cv=cv,
        scoring=scoring,
        refit='MCC',
        n_jobs=-1
    )

    svm_grid.fit(train_x, train_y)

    best_svm = svm_grid.best_estimator_

    y_pred = best_svm.predict(test_x)
    y_prob = best_svm.predict_proba(test_x)

    # Identify the probability column for death (class 1)
    death_index = list(best_svm.classes_).index(1)
    evaluate_metrics("SVM_FineTuned",test_y,y_pred,y_prob,death_index)
    return best_svm

# %%
def KNN_ft(train_x,train_y,test_x,test_y):
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ])

    knn_params = {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    }

    knn_grid = GridSearchCV(
        knn_pipeline,
        knn_params,
        cv=cv,
        scoring=scoring,
        refit='MCC',
        n_jobs=-1
    )

    knn_grid.fit(train_x, train_y)

    best_knn = knn_grid.best_estimator_

    y_pred = best_knn.predict(test_x)
    # Identify the probability column for death (class 1)
    death_index = list(best_knn.classes_).index(1)
    y_prob = best_knn.predict_proba(test_x) 
    evaluate_metrics("KNN_FineTuned",test_y,y_pred,y_prob,death_index)
    return best_knn


# %%
def NaiveBayes_ft(train_x,train_y,test_x,test_y):
    nb_model = GaussianNB()

    # Applied smoothing technique to fine tune hyper parameter
    nb_params = {
        'var_smoothing': np.logspace(-12, -6, 7)
    }

    nb_grid = GridSearchCV(
        estimator=nb_model,
        param_grid=nb_params,
        cv=cv,
        scoring=scoring,
        refit='MCC',
        n_jobs=-1
    )

    nb_grid.fit(train_x, train_y)

    best_nb = nb_grid.best_estimator_

    y_pred = best_nb.predict(test_x)
    # Identify the probability column for death (class 1)
    death_index = list(best_nb.classes_).index(1)
    y_prob = best_nb.predict_proba(test_x) 
    evaluate_metrics("NaiveBayes_FineTuned",test_y,y_pred,y_prob,death_index)
    return best_nb

# %%

def evaluate_metrics(model_name,test_y,y_pred,y_prob,death_index):
    # Evaluating metrics
    var_accuracy = accuracy_score(test_y, y_pred)

    var_precision = precision_score(
        test_y, y_pred, pos_label=1
        
    )

    var_recall = recall_score(
        test_y, y_pred, pos_label=1
    )

    var_f1_score = f1_score(
        test_y, y_pred, pos_label=1
    )

    var_auc = roc_auc_score(
        (test_y == 1).astype(int), y_prob[:, death_index]
    )

    var_mcc = matthews_corrcoef(test_y, y_pred)

    print(model_name,var_accuracy,var_auc,var_f1_score,var_mcc,var_precision,var_recall)
    # Append results to the collection
    results.append({
        'Model': model_name,
        'Accuracy': float(var_accuracy),
        'F1 Score': float(var_f1_score),
        'Precision': float(var_precision),
        'Recall': float(var_recall),
        'AUC': float(var_auc),
        'MCC': float(var_mcc)
    })
    
    # Classification Report
    report = classification_report(
        test_y,
        y_pred,
        labels=[1, 2],
        target_names=["Died", "Survived"],
        output_dict=True,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(
        test_y,
        y_pred,
        labels=[1, 2]
    )
    
    classification_reports.append({
    "Model": model_name,
    "Report": report
    })

    confusion_matrices.append({
        "Model": model_name,
        "Matrix": cm
    })

# Function to save each model
def save_model(model, filename):
    os.makedirs("model", exist_ok=True)
    file_path = os.path.join("model", filename)
    joblib.dump(model, file_path)
    print(f"Model saved successfully: {file_path}")

def load_model(filename):
    file_path = os.path.join("model", filename)
    model = joblib.load(file_path)
    return model

loaded_model = load_model("logistic_regression.joblib")
print("Model loaded successfully!")
print(loaded_model)

# %%
def apply_model(model_name, train_x, train_y, test_x, test_y,
                train_x_scaled, test_x_scaled):
    
    if model_name == 'Logistic':
        # calling both regular and fine tuned model for logistic
        baseline_model =  logistic_reg(train_x_scaled, train_y, test_x_scaled, test_y)
        fine_tuned_model = Logistic_ft(train_x, train_y, test_x, test_y)
        # Calling save model
        save_model(fine_tuned_model,
                "logistic_regression.joblib")
        return baseline_model,fine_tuned_model
    elif model_name == 'DecisionTree':
        baseline_model = decision_tree(train_x, train_y, test_x, test_y)
        fine_tuned_model = DecisionTree_ft(train_x, train_y, test_x, test_y)
        # Calling save model
        save_model(fine_tuned_model,
                "decision_tree.joblib")
        return baseline_model,fine_tuned_model
    elif model_name == 'SVM':
        baseline_model = SVM(train_x_scaled, train_y, test_x_scaled, test_y)
        fine_tuned_model = SVM_ft(train_x, train_y, test_x, test_y)
        # Calling save model
        save_model(fine_tuned_model,
                "svm.joblib")
        return baseline_model,fine_tuned_model
    elif model_name == 'NaiveBayes':
        baseline_model = NaiveBayes(train_x, train_y, test_x, test_y)
        fine_tuned_model = NaiveBayes_ft(train_x, train_y, test_x, test_y)
        # Calling save model
        save_model(fine_tuned_model,
                "naivebayes.joblib")
        return baseline_model,fine_tuned_model
    elif model_name == 'KNN':
        baseline_model = KNN(train_x_scaled, train_y, test_x_scaled, test_y)
        fine_tuned_model = KNN_ft(train_x, train_y, test_x, test_y)
        # Calling save model
        save_model(fine_tuned_model,
                   "knn.joblib")
        return baseline_model,fine_tuned_model
    elif model_name == 'Logistic_ft':
        return Logistic_ft(train_x, train_y, test_x, test_y)
    elif model_name == 'DecisionTree_ft':
        return DecisionTree_ft(train_x, train_y, test_x, test_y)
    elif model_name == 'SVM_ft':
        return SVM_ft(train_x, train_y, test_x, test_y)
    elif model_name == 'NaiveBayes_ft':
        return NaiveBayes_ft(train_x, train_y, test_x, test_y)
    elif model_name == 'KNN_ft':
        return KNN_ft(train_x, train_y, test_x, test_y)
    else:
        raise ValueError("Unsupported model")

# %%
def print_results():
    results_df = pd.DataFrame(results)
    print(
        f"{'Model':<20} "
        f"{'Accuracy':>12} "
        f"{'F1 Score':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'AUC':>12} "
        f"{'MCC':>12}"
    )

    print("-" * 104)
    
    for index, row in results_df.iterrows():
        print(
            f"{row['Model']:<20} "
            f"{row['Accuracy']:>12.6f} "
            f"{row['F1 Score']:>12.6f} "
            f"{row['Precision']:>12.6f} "
            f"{row['Recall']:>12.6f} "
            f"{row['AUC']:>12.6f} "
            f"{row['MCC']:>12.6f}"
        )
    return results_df