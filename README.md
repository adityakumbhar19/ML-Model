# Alzheimer's Disease Detection ML Module

A comprehensive machine learning project for detecting Alzheimer's disease based on patient symptoms.

## 📋 Project Overview

This ML module uses a Random Forest Classifier to predict whether a patient is suffering from Alzheimer's disease based on various symptoms and medical indicators.

## 📊 Dataset

The project includes a **hardcoded dataset** with 40 patient records containing:
- **Memory Loss** (0-10 severity scale)
- **Confusion Level** (0-10 severity scale)
- **Difficulty Speaking** (0-10 severity scale)
- **Mood Changes** (0-10 severity scale)
- **Sleep Disorders** (0-10 severity scale)
- **Loss of Coordination** (0-10 severity scale)
- **Age** (in years)
- **Target**: Binary classification (1 = Alzheimer's, 0 = Healthy)

The dataset is balanced with equal representation of Alzheimer's patients and healthy controls.

## 🏗️ Project Structure

```
ML-Module/
├── data/
│   └── alzheimers_data.py          # Hardcoded patient symptom data
├── models/                          # Saved trained models
│   ├── alzheimers_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl                  # Feature scaler
│   ├── evaluation_metrics.png      # Performance visualizations
│   └── feature_importance.png      # Feature importance chart
├── train_model.py                  # Main training script
├── predict.py                      # Prediction on new patients
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── __init__.py                     # Package initialization
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load the hardcoded dataset
- Split data into training (80%) and testing (20%) sets
- Train a Random Forest Classifier
- Display comprehensive evaluation metrics
- Generate visualization plots
- Save the trained model and scaler

### 3. Make Predictions

```bash
python predict.py
```

This will:
- Load the previously trained model
- Make predictions on sample patient data
- Display prediction results with confidence scores

## 📈 Model Performance

The trained model provides:
- **Accuracy**: Classification accuracy on test set
- **Precision**: False positive rate control
- **Recall**: Ability to identify actual Alzheimer's cases
- **F1-Score**: Balanced performance metric
- **AUC-ROC**: Overall discrimination ability

## 🔧 Key Features

### AlzheimersDetectionModel Class

```python
# Initialize
model = AlzheimersDetectionModel()

# Train the model
model.train()

# Make predictions
predictions, probabilities = model.predict(new_patient_data)

# Make predictions for individual patients
model.predict_patient(patient_dataframe)

# Save and load models
model.save_model()
loaded_model, scaler = AlzheimersDetectionModel.load_model()
```

## 📝 Example Usage

```python
from train_model import AlzheimersDetectionModel
import pandas as pd

# Initialize and train
model = AlzheimersDetectionModel()
model.train()

# Create new patient data
patient = pd.DataFrame({
    'memory_loss': [8],
    'confusion_level': [7],
    'difficulty_speaking': [6],
    'mood_changes': [5],
    'sleep_disorders': [7],
    'loss_of_coordination': [6],
    'age': [82]
})

# Make prediction
model.predict_patient(patient)
```

## 📊 Output Files

After training, the following files are generated:

1. **models/alzheimers_model.pkl** - Serialized trained model
2. **models/scaler.pkl** - Feature scaler for standardization
3. **models/evaluation_metrics.png** - Confusion matrix and ROC curve
4. **models/feature_importance.png** - Feature importance visualization

## 🎯 Prediction Output

The model provides:
- Binary prediction (Positive/Negative for Alzheimer's)
- Confidence score (probability percentage)
- Risk assessment based on symptoms

## ⚙️ Model Details

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 10
- **Random State**: 42 (for reproducibility)
- **Feature Scaling**: StandardScaler

## 📚 Technologies Used

- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 🔬 How It Works

1. **Data Loading**: Hardcoded patient symptoms are loaded
2. **Preprocessing**: Features are standardized using StandardScaler
3. **Model Training**: Random Forest learns patterns from training data
4. **Evaluation**: Performance is measured using multiple metrics
5. **Prediction**: New patients are classified as Alzheimer's or healthy

## ⚠️ Medical Disclaimer

This is a **demonstration ML project** for educational purposes. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for actual medical diagnosis and treatment decisions.

## 🔐 Data Privacy

All data in this project is **hardcoded synthetic data** for demonstration purposes only. No real patient data is used.

## 📞 Support

For questions or improvements, please refer to the code documentation and comments within each module.

---

**Last Updated**: January 2026
**Version**: 1.0
