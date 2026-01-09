# Hardcoded Alzheimer's Disease Patient Symptom Data
# This dataset contains various symptoms and indicators for Alzheimer's disease detection

import pandas as pd

def get_alzheimers_data():
    """
    Returns a dataset with Alzheimer's patient symptoms.
    
    Features:
    - Memory Loss (0-10)
    - Confusion Level (0-10)
    - Difficulty Speaking (0-10)
    - Mood Changes (0-10)
    - Sleep Disorders (0-10)
    - Loss of Coordination (0-10)
    - Age (in years)
    
    Target:
    - 1 = Alzheimer's Present
    - 0 = No Alzheimer's
    """
    
    data = {
        'memory_loss': [8, 2, 7, 1, 9, 3, 6, 2, 8, 4, 7, 5, 9, 2, 8, 1, 6, 3, 7, 5,
                       9, 2, 8, 4, 7, 1, 9, 3, 6, 2, 8, 5, 7, 3, 9, 1, 8, 4, 6, 2],
        
        'confusion_level': [7, 1, 8, 2, 9, 2, 7, 1, 8, 3, 6, 2, 8, 1, 7, 2, 6, 3, 7, 1,
                           9, 2, 8, 3, 7, 1, 9, 2, 6, 1, 8, 3, 7, 2, 9, 1, 8, 2, 6, 3],
        
        'difficulty_speaking': [6, 2, 7, 1, 8, 2, 6, 1, 7, 3, 5, 2, 7, 1, 6, 2, 5, 3, 6, 1,
                               8, 2, 7, 3, 6, 1, 8, 2, 5, 1, 7, 3, 6, 2, 8, 1, 7, 2, 5, 3],
        
        'mood_changes': [5, 2, 6, 1, 7, 2, 5, 1, 6, 3, 4, 2, 6, 1, 5, 2, 4, 3, 5, 1,
                        7, 2, 6, 3, 5, 1, 7, 2, 4, 1, 6, 3, 5, 2, 7, 1, 6, 2, 4, 3],
        
        'sleep_disorders': [7, 1, 8, 2, 9, 1, 7, 2, 8, 3, 6, 1, 8, 2, 7, 1, 6, 3, 7, 2,
                           9, 1, 8, 3, 7, 2, 9, 1, 6, 2, 8, 3, 7, 1, 9, 2, 8, 1, 6, 3],
        
        'loss_of_coordination': [6, 1, 7, 2, 8, 2, 6, 1, 7, 3, 5, 2, 7, 1, 6, 2, 5, 3, 6, 2,
                                8, 1, 7, 3, 6, 2, 8, 1, 5, 2, 7, 3, 6, 1, 8, 2, 7, 1, 5, 3],
        
        'age': [78, 45, 82, 52, 85, 48, 80, 55, 83, 50, 79, 47, 84, 49, 81, 46, 77, 51, 79, 48,
               86, 44, 82, 53, 78, 47, 85, 50, 76, 45, 81, 52, 79, 49, 84, 46, 82, 51, 77, 54],
        
        'alzheimers': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    return df


def get_sample_predictions():
    """
    Returns sample patient data for testing predictions.
    """
    sample_patients = {
        'memory_loss': [8, 2, 7],
        'confusion_level': [7, 2, 6],
        'difficulty_speaking': [6, 1, 5],
        'mood_changes': [5, 1, 4],
        'sleep_disorders': [7, 2, 6],
        'loss_of_coordination': [6, 1, 5],
        'age': [82, 50, 78]
    }
    
    df = pd.DataFrame(sample_patients)
    return df


if __name__ == "__main__":
    # Display the dataset
    data = get_alzheimers_data()
    print("Alzheimer's Disease Patient Dataset:")
    print(data.head(10))
    print(f"\nDataset shape: {data.shape}")
    print(f"\nColumn names: {data.columns.tolist()}")
    print(f"\nAlzheimer's distribution:\n{data['alzheimers'].value_counts()}")
