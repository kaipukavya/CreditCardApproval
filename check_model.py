
import joblib

model = joblib.load("credit_card_model.pkl")

print("Model type:", type(model))

if hasattr(model, "feature_names_in_"):
    print("\nExpected features:")
    print(list(model.feature_names_in_))

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(name, type(step).__name__)

    for name, step in model.named_steps.items():
        if hasattr(step, "transformers_"):
            print("\nColumnTransformer:", name)
            for transformer_name, transformer, columns in step.transformers_:
                print(transformer_name, ":", list(columns))
