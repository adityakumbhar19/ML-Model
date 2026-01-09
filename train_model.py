"""
Alzheimer's Disease Detection ML Model
This script trains a machine learning model to detect Alzheimer's disease
based on patient symptoms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from data.alzheimers_data import get_alzheimers_data, get_sample_predictions


class AlzheimersDetectionModel:
    """
    Machine Learning model for Alzheimer's disease detection.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = None
        
    def load_data(self):
        """Load the Alzheimer's disease dataset."""
        print("Loading Alzheimer's disease dataset...")
        df = get_alzheimers_data()
        
        # Separate features and target
        X = df.drop('alzheimers', axis=1)
        y = df['alzheimers']
        
        self.feature_names = X.columns.tolist()
        print(f"Features: {self.feature_names}")
        print(f"Dataset size: {len(X)} samples")
        print(f"Alzheimer's cases: {(y == 1).sum()}")
        print(f"Healthy cases: {(y == 0).sum()}")
        
        return X, y
    
    def train(self, test_size=0.2, random_state=42):
        """Train the ML model."""
        print("\n" + "="*60)
        print("TRAINING THE MODEL")
        print("="*60)
        
        # Load data
        X, y = self.load_data()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        print("Training Random Forest Classifier...")
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate the model
        self.evaluate(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        self.show_feature_importance()
        
        return self.model, self.scaler
    
    def evaluate(self, y_test, y_pred, y_pred_proba):
        """Evaluate model performance."""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Healthy', 'Alzheimer\'s']))
        
        # Create visualizations
        self.plot_metrics(y_test, y_pred, y_pred_proba)
    
    def plot_metrics(self, y_test, y_pred, y_pred_proba):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        print("\nEvaluation metrics plot saved to: models/evaluation_metrics.png")
        plt.close()
    
    def show_feature_importance(self):
        """Display feature importance."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(feature_importance_df.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance for Alzheimer\'s Detection')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to: models/feature_importance.png")
        plt.close()
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_patient(self, patient_data):
        """Predict for a single patient and display results."""
        prediction, probability = self.predict(patient_data)
        
        patient_df = pd.DataFrame(patient_data)
        
        for idx in range(len(patient_df)):
            print(f"\nPatient {idx + 1} Prediction:")
            print("-" * 50)
            for col, val in zip(self.feature_names, patient_data.iloc[idx]):
                print(f"  {col}: {val}")
            
            pred = prediction[idx]
            prob = probability[idx]
            
            if pred == 1:
                print(f"\n  ⚠️  POSITIVE: Likely has Alzheimer's disease")
                print(f"  Confidence: {prob[1]*100:.2f}%")
            else:
                print(f"\n  ✓ NEGATIVE: Unlikely to have Alzheimer's disease")
                print(f"  Confidence: {prob[0]*100:.2f}%")
    
    def save_model(self, model_path='models/alzheimers_model.pkl', 
                   scaler_path='models/scaler.pkl'):
        """Save the trained model and scaler."""
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
    
    @staticmethod
    def load_model(model_path='models/alzheimers_model.pkl',
                   scaler_path='models/scaler.pkl'):
        """Load a saved model and scaler."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")
        
        return model, scaler


def main():
    """Main function to train and test the model."""
    
    # Create and train the model
    alzheimers_model = AlzheimersDetectionModel()
    model, scaler = alzheimers_model.train()
    
    # Save the model
    print("\n" + "="*60)
    print("SAVING THE MODEL")
    print("="*60)
    alzheimers_model.save_model()
    
    # Make predictions on sample patients
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE PATIENTS")
    print("="*60)
    sample_data = get_sample_predictions()
    alzheimers_model.predict_patient(sample_data)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
