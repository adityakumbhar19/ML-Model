"""
Prediction script for testing the trained Alzheimer's detection model.
This script loads the saved model and makes predictions on new patient data.
"""

import pandas as pd
from train_model import AlzheimersDetectionModel


def get_custom_patient_data():
    """Create custom patient data for prediction."""
    patients = {
        'memory_loss': [8, 3, 9, 2],
        'confusion_level': [7, 2, 8, 1],
        'difficulty_speaking': [6, 1, 7, 2],
        'mood_changes': [5, 2, 6, 1],
        'sleep_disorders': [7, 1, 8, 2],
        'loss_of_coordination': [6, 2, 7, 1],
        'age': [82, 55, 85, 48]
    }
    
    return pd.DataFrame(patients)


def main():
    """Load model and make predictions."""
    
    print("="*60)
    print("ALZHEIMER'S DETECTION - PREDICTION MODULE")
    print("="*60)
    
    # Initialize model
    model_detector = AlzheimersDetectionModel()
    
    # Get custom patient data
    print("\nLoading custom patient data...")
    patient_data = get_custom_patient_data()
    
    # Load trained model
    print("Loading trained model...")
    model_detector.model, model_detector.scaler = AlzheimersDetectionModel.load_model()
    model_detector.trained = True
    model_detector.feature_names = patient_data.columns.tolist()
    
    # Make predictions
    print("\nMaking predictions...")
    model_detector.predict_patient(patient_data)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
