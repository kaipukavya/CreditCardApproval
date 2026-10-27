## 🚀 Live Demo
[Click here to view the deployed project](https://creditcardapproval-hw6a.onrender.com/)
# Credit Card Approval Prediction Using XGBoost

## Project Overview

The **Credit Card Approval Prediction System** is a Machine Learning project that predicts whether a credit card application is likely to be approved or rejected based on applicant information.

The project uses the **XGBoost classification algorithm** to make predictions and a **Flask web application** to provide a simple user interface. The application also displays a confidence percentage for the prediction.

## Objectives

- Predict credit card application approval using Machine Learning.
- Analyze applicant details and credit-related information.
- Compare different classification algorithms.
- Develop a user-friendly web application using Flask.
- Display prediction results with a model confidence score.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Model Saving:** Joblib
- **Development Environment:** Visual Studio Code

## Machine Learning Algorithms

The project includes the following classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

XGBoost is used as the prediction model in the web application.

## Project Structure

```text
CreditCardApproval/
│
├── data/
│   └── dataset.csv
│
├── templates/
│   ├── .gitignore
│   └── index.html
│
├── app.py
├── check_data.py
├── check_model.py
├── data_analysis.py
├── preprocessing.py
├── logistic_regression.py
├── decision_tree.py
├── random_forest.py
├── xgboost_model.py
├── save_model.py
├── credit_card_model.pkl
├── requirements.txt
└── README.md
```

## Dataset

The project uses a credit approval dataset containing applicant-related attributes, such as:

- Gender
- Age
- Debt
- Marital status
- Bank customer status
- Industry category
- Ethnicity category
- Years employed
- Previous default status
- Employment status
- Credit score
- Driver's license status
- Citizenship category
- ZIP code category
- Income

The target column is **`Approved`**, which represents the credit application outcome.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kaipukavya/CreditCardApproval.git
```

### 2. Open the Project Folder

```bash
cd CreditCardApproval
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Flask application using:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Enter the applicant details in the form and click the prediction button to view the result.

## How It Works

1. Applicant details are entered through the web form.
2. The input data is prepared for the trained model.
3. The saved XGBoost model processes the input.
4. The system predicts whether the application is likely to be approved or rejected.
5. The application displays the prediction and model confidence percentage.

## Model Evaluation

The classification algorithms can be evaluated using accuracy, confusion matrices, precision, recall, and F1-score.

The evaluation results depend on the dataset split and model configuration. Model confidence percentages are estimates and do not guarantee that a prediction is correct.

## Future Enhancements

- Improve model performance through hyperparameter tuning.
- Add more evaluation metrics and visualizations.
- Improve input validation and error handling.
- Deploy the application to a cloud platform.
- Add further model interpretability features.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions should not be used as the sole basis for real financial or credit approval decisions.

## Author

**Kavya Kaipu**

GitHub: [kaipukavya](https://github.com/kaipukavya)
