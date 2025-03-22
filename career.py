import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample dataset (You can replace it with a real dataset)
data = {
    'Stream': ['Science', 'Commerce', 'Arts', 'Science', 'Arts', 'Commerce'],
    'Interest': ['Engineering', 'Finance', 'Design', 'Medical', 'History', 'Business'],
    'Career': ['Software Engineer', 'Accountant', 'Graphic Designer', 'Doctor', 'Historian', 'Entrepreneur']
}

df = pd.DataFrame(data)

# Encoding categorical data
le_stream = LabelEncoder()
le_interest = LabelEncoder()
le_career = LabelEncoder()

df['Stream'] = le_stream.fit_transform(df['Stream'])
df['Interest'] = le_interest.fit_transform(df['Interest'])
df['Career'] = le_career.fit_transform(df['Career'])

# Splitting data
X = df[['Stream', 'Interest']]
y = df['Career']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
with open('career_model.pkl', 'wb') as file:
    pickle.dump((model, le_stream, le_interest, le_career), file)

print("Model Trained and Saved!")
