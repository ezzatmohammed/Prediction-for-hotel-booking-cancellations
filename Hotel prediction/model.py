import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split ,GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from imblearn.over_sampling import RandomOverSampler
import pickle





#read dataset 
df=pd.read_csv("ml_data.csv")

# Separate features and target variable
X=df[['number of week nights', 'lead time', 'average price','special requests', 'day of reservation', 'month of reservation']]
y=df["booking status"]


# Initialize the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Resample the dataset
X_resampled, y_resampled = ros.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)







# Create a pipeline
pipeline = Pipeline([
    ("standard scaler" ,StandardScaler() ),                
    ('hyperparameter_tuning', RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions={
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        n_iter=50,
        cv=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)


# Save the trained pipeline to a file named 'model.pkl'
pickle.dump(pipeline, open('model.pkl', 'wb'))


