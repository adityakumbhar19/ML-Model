"""
Model evaluation and analysis script.
Provides detailed analysis of the trained model.
"""

import pandas as pd
from train_model import AlzheimersDetectionModel
from data.alzheimers_data import get_alzheimers_data
from sklearn.model_selection import train_test_split


def analyze_model():
    """Analyze the trained model in detail."""
    
    print("="*70)
    print("DETAILED MODEL ANALYSIS")
    print("="*70)
    
    # Load data
    df = get_alzheimers_data()
    X = df.drop('alzheimers', axis=1)
    y = df['alzheimers']
    
    # Train model
    model_detector = AlzheimersDetectionModel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale and train
    X_train_scaled = model_detector.scaler.fit_transform(X_train)
    X_test_scaled = model_detector.scaler.transform(X_test)
    model_detector.model.fit(X_train_scaled, y_train)
    model_detector.trained = True
    model_detector.feature_names = X.columns.tolist()
    
    # Get predictions
    y_pred = model_detector.model.predict(X_test_scaled)
    y_pred_proba = model_detector.model.predict_proba(X_test_scaled)
    
    # Display detailed results
    print("\n📊 PREDICTION RESULTS ON TEST SET:")
    print("-" * 70)
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Healthy_Prob': y_pred_proba[:, 0],
        'Alzheimers_Prob': y_pred_proba[:, 1],
        'Correct': y_test.values == y_pred
    })
    print(results_df.to_string(index=False))
    
    print("\n📈 PREDICTION ACCURACY:")
    print("-" * 70)
    correct = (y_test.values == y_pred).sum()
    total = len(y_test)
    print(f"Correct Predictions: {correct}/{total} ({correct/total*100:.2f}%)")
    
    print("\n🎯 DISEASE DETECTION ANALYSIS:")
    print("-" * 70)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    
    print(f"True Positives (Correctly identified Alzheimer's): {tp}")
    print(f"True Negatives (Correctly identified Healthy): {tn}")
    print(f"False Positives (Healthy marked as Alzheimer's): {fp}")
    print(f"False Negatives (Alzheimer's marked as Healthy): {fn}")
    
    print("\n⚠️  CLINICAL SIGNIFICANCE:")
    print("-" * 70)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Positive Predictive Value: {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    analyze_model()
