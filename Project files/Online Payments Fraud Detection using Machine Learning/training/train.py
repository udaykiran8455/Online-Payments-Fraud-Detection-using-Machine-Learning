import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset (Update the path to your .csv file)
df = pd.read_csv("payments_data.csv") 

# 2. Data Pre-processing
# Removing non-numeric or unnecessary columns
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedAvg'], axis=1, errors='ignore')

# Handle Categorical Values (Transaction Type)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# 3. Define Features (X) and Target (y)
# Features used: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# 4. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Save the model as .pkl
pickle.dump(model, open('payments.pkl', 'wb'))
print("Model saved as payments.pkl!")