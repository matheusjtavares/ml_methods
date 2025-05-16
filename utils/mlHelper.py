
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


class mlHelper():
    '''This class contains helper methods to support ML analysis '''
    def __init__(self):
        pass
    
    def create_sklearn_vectors(self, data:pd.DataFrame, y_labels:list, x_labels:list = None):
        
        y = data[y_labels]
        if x_labels:
            X = data[x_labels]
        else:
            X = data[~y_labels]
        return X,y
    def create_tensorflow_vectors(self, data:pd.DataFrame, y_labels:list, x_labels:list = None):
        return
    def create_dummies_with_cut(self,data:pd.DataFrame, dummy_settings:dict):
        for column,settings in dummy_settings.items():
            data[f'{column}Groups'] = pd.cut(data[column], bins = settings['bins'], include_lowest=True, labels=settings['labels'])
            data = pd.get_dummies(data,columns=[f'{column}Groups'])
        return data
    
    def get_ml_metrics(self,y:list,y_predicted:list):
        precision,recall,fscore,support  = precision_recall_fscore_support(y,y_predicted)
        return {
            'Precision':precision[0],
            'Recall':recall[0],
            'Fscore':fscore[0],
            'Support':support[0],
                }
    
    def plot_correlation_matrix(self,data:pd.DataFrame,columns:list) -> pd.DataFrame:
        corrMatrix = data[columns].corr()
        plt.figure(figsize=(12,12))
        plt.title('Correlation of Features', y=1.05, size=15)
        sn.heatmap(corrMatrix.astype(float),linewidths=0.1,vmax=1.0, square=True, cmap='viridis', linecolor='white', annot=True,fmt='.2f')
        plt.show()
        plt.close()
        return corrMatrix
    
    def plot_confusion_matrix(self,y:list,y_predicted:list,file_path:str=None)-> np.array
        fig, ax = plt.subplots(figsize=(8, 4))
        cm = confusion_matrix(y, y_predicted)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        ax.set_title('Training Samples')
        # Customizations
        plt.tight_layout()
        ml_metrics = self.get_ml_metrics(y,y_predicted)
        ax.text(
            x=-1,  # x-coordinate of the text
            y=2,  # y-coordinate of the text
            s=f'''Metrics:\nPrecision: {ml_metrics['Precision']*100:.2f} %\nRecall: {ml_metrics['Recall']*100:.2f} %\nF1: {ml_metrics['Fscore']*100:.2f} %''',  # Text string
            fontsize=8,  # Font size
            color="black",  # Text color
            bbox={
                "facecolor": "lightgray",  # Background color of the box
                "edgecolor": "black",  # Border color of the box
                "boxstyle": "round,pad=0.5",  # Box style (e.g., round, square) and padding
                "alpha": 0.5,  # Transparency of the box
            },
        )
        # Customizations
        plt.tight_layout()
        return cm