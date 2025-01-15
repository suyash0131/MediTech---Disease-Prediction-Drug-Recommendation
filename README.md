# MediTech - Disease Prediction & Drug Recommendation

## Project Description
MediTech is an AI-driven healthcare solution designed to assist in disease prediction and recommend suitable drugs based on symptoms. Leveraging machine learning algorithms and user-friendly interfaces, the project provides a powerful tool for early diagnosis and treatment suggestions.

---

## Features
- **Disease Prediction**: Predicts diseases based on user-provided symptoms.
- **Drug Recommendation**: Suggests appropriate drugs for diagnosed conditions.
- **Interactive User Interface**: Easy-to-use web interface for healthcare professionals and patients.
- **Customizable Models**: Supports retraining to accommodate new diseases and drugs.

---

## Folder Structure

```
MediTech---Disease-Prediction-Drug-Recommendation/
|
|-- MediTech/               # Core machine learning logic
|-- predictions/            # Model outputs and prediction-related scripts
|-- db.sqlite3              # SQLite database for Django
|-- manage.py               # Django's management script
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Django 3.2+
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/suyash0131/MediTech---Disease-Prediction-Drug-Recommendation.git
   cd MediTech---Disease-Prediction-Drug-Recommendation
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Apply database migrations:
   ```bash
   python manage.py migrate
   ```
5. Run the development server:
   ```bash
   python manage.py runserver
   ```
6. Access the application at `http://127.0.0.1:8000/`.

---

## Usage
- Enter symptoms in the web interface.
- View predicted disease and recommended medications.

---
