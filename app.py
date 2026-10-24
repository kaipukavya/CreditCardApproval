
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("credit_card_model.pkl")

# Features expected by the model
fields = [
    "Gender",
    "Age",
    "Debt",
    "Married",
    "BankCustomer",
    "Industry",
    "Ethnicity",
    "YearsEmployed",
    "PriorDefault",
    "Employed",
    "CreditScore",
    "DriversLicense",
    "Citizen",
    "ZipCode",
    "Income"
]

# Convert form values to the numeric format used during training.
# Verify these mappings against your original training dataset.
category_mapping = {
    "Gender": {"a": 1, "b": 0},
    "Married": {"y": 1, "u": 0},
    "BankCustomer": {"g": 1, "p": 0},
    "PriorDefault": {"t": 1, "f": 0},
    "Employed": {"t": 1, "f": 0},
    "DriversLicense": {"t": 1, "f": 0}
}

numeric_fields = [
    "Age",
    "Debt",
    "YearsEmployed",
    "CreditScore",
    "Income"
]


@app.route("/")
def home():
    return render_template("index.html", fields=fields)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        applicant = {}

        # Read values submitted through the form
        for field in fields:
            value = request.form.get(field, "").strip()

            if not value:
                return render_template(
                    "index.html",
                    fields=fields,
                    error=f"Please enter {field}."
                )

            # Convert categorical form values when required
            if field in category_mapping:
                if value not in category_mapping[field]:
                    raise ValueError(f"Invalid value for {field}.")
                value = category_mapping[field][value]

            applicant[field] = value

        # Convert numerical inputs
        for field in numeric_fields:
            applicant[field] = float(applicant[field])

        # Keep ZipCode as text because your pipeline treats it
        # as a categorical feature.
        applicant["ZipCode"] = str(applicant["ZipCode"])

        # Arrange features in the same order as training
        input_data = pd.DataFrame([applicant], columns=fields)

        # Predict approval/rejection
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            result = "Likely to be approved"
        else:
            result = "Likely to be rejected"

        # Calculate confidence if the model supports probabilities
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            # Find the probability associated with the predicted class
            predicted_index = classes.index(prediction)
            confidence = round(
                float(probabilities[predicted_index]) * 100, 2
            )

        return render_template(
            "index.html",
            fields=fields,
            result=result,
            confidence=confidence
        )

    except ValueError as error:
        return render_template(
            "index.html",
            fields=fields,
            error=str(error)
        )

    except Exception:
        return render_template(
            "index.html",
            fields=fields,
            error=(
                "Unable to make a prediction. "
                "Please check your input values and model configuration."
            )
        )


if __name__ == "__main__":
    app.run(debug=True)
