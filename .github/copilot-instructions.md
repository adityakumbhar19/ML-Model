# Alzheimer's Disease Detection ML Module

## Project Overview
A comprehensive machine learning project for detecting Alzheimer's disease based on patient symptoms using hardcoded synthetic data.

## Project Structure
- `data/alzheimers_data.py` - Hardcoded patient symptom dataset (40 patients)
- `train_model.py` - Main training script with AlzheimersDetectionModel class
- `predict.py` - Prediction script for new patient data
- `analyze.py` - Detailed model analysis and evaluation
- `requirements.txt` - Python dependencies
- `models/` - Directory for saved models and visualizations

## Key Features
- **Hardcoded Dataset**: 40 patient records with 7 features (symptoms + age) and binary classification
- **Model**: Random Forest Classifier with 100 trees
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Visualizations**: Confusion matrix, ROC curve, feature importance
- **Predictions**: Classification with confidence scores

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python train_model.py
```

### Make Predictions
```bash
python predict.py
```

### Analyze Model
```bash
python analyze.py
```

## Dataset Details
Features:
- Memory Loss (0-10 scale)
- Confusion Level (0-10 scale)
- Difficulty Speaking (0-10 scale)
- Mood Changes (0-10 scale)
- Sleep Disorders (0-10 scale)
- Loss of Coordination (0-10 scale)
- Age (years)

Target: Binary (1 = Alzheimer's, 0 = Healthy)

## Technologies
- scikit-learn, pandas, numpy, matplotlib, seaborn

## Model Performance
Provides: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix, Feature Importance
