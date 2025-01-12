import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Optional

class StanceClassifierEvaluator:
    """
    A class to evaluate stance classification models with detailed metrics and visualizations.
    """
    def __init__(self, class_labels: Optional[List[str]] = None):
        self.class_labels = class_labels
    
        
    def evaluate(self, y_true: Union[List, np.ndarray], 
                y_pred: Union[List, np.ndarray]) -> dict:
        # Convert inputs to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get classification report with detailed metrics
        report = classification_report(y_true, y_pred, 
                                    target_names=self.class_labels,
                                    output_dict=True)
        
        # Calculate overall F1 score
        overall_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate per-class F1 scores
        per_class_f1 = {}
        if self.class_labels:
            for i, label in enumerate(self.class_labels):
                mask = (y_true == i)
                per_class_f1[label] = f1_score(y_true == i, y_pred == i)
        else:
            unique_labels = sorted(set(y_true))
            for i in unique_labels:
                per_class_f1[f'Class {i}'] = f1_score(y_true == i, y_pred == i)
                
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'overall_f1': overall_f1,
            'per_class_f1': per_class_f1
        }
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            figsize: tuple = (10, 8)) -> None:
        plt.figure(figsize=figsize)
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=self.class_labels,
                   yticklabels=self.class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def print_metrics(self, metrics: dict) -> None:
        print("=== Stance Classifier Evaluation Results ===\n")
        
        print("Overall F1 Score:", f"{metrics['overall_f1']:.4f}")
        
        print("\nPer-class F1 Scores:")
        for class_name, f1 in metrics['per_class_f1'].items():
            print(f"{class_name}: {f1:.4f}")
            
        print("\nDetailed Classification Report:")
        report = metrics['classification_report']
        # Print only the relevant metrics from the classification report
        for label, scores in report.items():
            if isinstance(scores, dict):
                print(f"\n{label}:")
                print(f"  Precision: {scores['precision']:.4f}")
                print(f"  Recall: {scores['recall']:.4f}")
                print(f"  F1-score: {scores['f1-score']:.4f}")
                print(f"  Support: {scores['support']}")