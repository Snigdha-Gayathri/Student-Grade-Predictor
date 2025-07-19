from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset and preprocess
def load_and_train_model():
    data = pd.read_csv("student-mat.csv")
    data = data.drop(['school', 'age'], axis=1)

    binary_map = {'yes': 1, 'no': 0}
    for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
        data[col] = data[col].map(binary_map)

    data['sex'] = data['sex'].map({'F': 1, 'M': 0})
    data['address'] = data['address'].map({'U': 1, 'R': 0})
    data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})
    job_map = {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}
    data['Mjob'] = data['Mjob'].map(job_map)
    data['Fjob'] = data['Fjob'].map(job_map)
    reason_map = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
    data['reason'] = data['reason'].map(reason_map)
    guardian_map = {'mother': 0, 'father': 1, 'other': 2}
    data['guardian'] = data['guardian'].map(guardian_map)

    data = data.select_dtypes(include=[np.number])
    X = data.drop(columns=['G3', 'GradeAvg']) if 'GradeAvg' in data.columns else data.drop(columns=['G3'])
    y = data['G3']

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns, data

# Load model and training column names
model, feature_order, dataset = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Required input features
        student = {
            'G1': int(data['G1']),
            'G2': int(data['G2']),
            'studytime': int(data['studytime']),
            'failures': int(data['failures']),
            'absences': int(data['absences']),
            'sex': 1 if data['sex'].lower() == 'f' else 0,
            'internet': 1 if data['internet'].lower() == 'yes' else 0,
            'higher': 1 if data['higher'].lower() == 'yes' else 0,
            'Fedu': int(data['Fedu']),
            'Medu': int(data['Medu'])
        }

        # Fill missing values with column means
        for col in feature_order:
            if col not in student:
                student[col] = dataset[col].mean()

        input_df = pd.DataFrame([student])[feature_order]
        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_grade': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
