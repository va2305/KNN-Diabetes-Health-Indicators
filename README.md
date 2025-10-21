# KNN Diabetes Health Indicators Project

## Overview
This project uses the **K-Nearest Neighbors (KNN)** classification algorithm to predict the likelihood of **diabetes** based on the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset.  
The goal is to build an accurate and interpretable model to help identify individuals at risk of diabetes.

## Dataset
**Source:** [CDC Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**Features include:**  
- BMI (Body Mass Index)  
- Smoking habits  
- Physical Activity  
- Sleep Time  
- Alcohol Consumption  
- General Health Rating  
- Age Category  

**Target variable:**  
- 0 → No Diabetes  
- 1 → Diabetes  

## Workflow
1. **Data Exploration:** Checked for missing values and visualized correlations.  
2. **Preprocessing:** Split data into train/test sets and scaled features.  
3. **Modeling:** Built KNN classifier, tuned `k` for best performance.  
4. **Evaluation:** Calculated Accuracy, Precision, Recall, F1-score, and plotted Confusion Matrix.  

## Results
- **Accuracy:** ~85–90%  
- **Precision:** High (few false positives)  
- **Recall:** Balanced (few false negatives)  
- **F1 Score:** Stable performance  

## Technologies Used
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- scikit-learn  

## Future Improvements
- Use GridSearchCV for hyperparameter tuning  
- Compare KNN with other models like Logistic Regression, Random Forest  
- Add feature importance analysis  
- Build an interactive dashboard for real-time prediction  

## Author
**Varsha Chauhan (va2305)**  
Python Developer | Data Science Enthusiast  
