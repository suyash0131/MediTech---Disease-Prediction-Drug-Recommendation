from django.shortcuts import render
import joblib
import torch
import torch.nn as nn
import numpy as np
from django.template.defaulttags import register

# Load the disease prediction model
disease_model_path = 'predictions/Notebook/disease_prediction_model.pkl'
disease_model = joblib.load(disease_model_path)

# Load the drug recommendation model
drug_model_path = 'predictions/Notebook/model.pt'

# Define the architecture of the drug recommendation model
class DrugRecommendationModel(nn.Module):
    def __init__(self):
        super(DrugRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(75160, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3264)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

drug_model = DrugRecommendationModel()
drug_model.load_state_dict(torch.load(drug_model_path, map_location=torch.device("cpu")))
drug_model.eval()

# Custom symptoms list
custom_symptoms = [
    "fever", "cough", "headache", "nausea", "fatigue", "chills", "sore throat", 
    "loss of smell", "loss of taste", "muscle pain", "diarrhea", "rash", 
    "shortness of breath", "chest pain", "dizziness", "vomiting", "congestion", 
    "runny nose", "sneezing", "stiff neck", "blurred vision", "weight loss"
]

# Define the transformation from 22 features to 75160 feature space
def transform_features(features):
    transformed = np.zeros((1, 75160))  # Adjust the dimensions as needed
    for i, value in enumerate(features):
        if value == 1:  # Indicating the symptom is present
            transformed[0, i] = 1  # Set the corresponding feature
    return transformed

# Custom template filter for ordinal numbers
@register.filter
def ordinal(value):
    if value % 100 // 10 != 1:
        if value % 10 == 1:
            return f"{value}st"
        elif value % 10 == 2:
            return f"{value}nd"
        elif value % 10 == 3:
            return f"{value}rd"
    return f"{value}th"

def predict_disease(request):
    prediction = None

    if request.method == "POST":
        # Get the symptoms submitted from the form
        symptoms = [
            request.POST.get(f"symptom{i}") for i in range(1, 6)
        ]

        # One-hot encode the symptoms
        input_features = [1 if symptom in symptoms else 0 for symptom in custom_symptoms]

        # Transform input features to match the expected input size of the models
        transformed_features = transform_features(input_features)

        try:
            # Predict disease
            disease_prediction = disease_model.predict([input_features])[0]

            # Predict drug recommendation
            input_tensor = torch.tensor(transformed_features, dtype=torch.float32)
            drug_recommendation = drug_model(input_tensor).detach().numpy()
            recommended_drugs = [f"Drug {i}" for i in np.argmax(drug_recommendation, axis=1)]
            
            # Combine predictions
            prediction = f"Disease: {disease_prediction}, Recommended Drugs: {', '.join(recommended_drugs)}"
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

    return render(request, "predictions/predict.html", {
        "range": range(1, 6),  # Five symptoms
        "symptom_choices": custom_symptoms,  # List of symptoms
        "prediction": prediction,  # Result from the model
    })
