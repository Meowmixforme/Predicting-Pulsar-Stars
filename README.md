# Pulsar Detection Using Machine Learning

## Overview
An advanced machine learning project focused on automated pulsar detection in radio astronomy data. The project implements and compares multiple classification algorithms to address the challenge of identifying pulsars among large volumes of radio telescope data with high false-positive rates.

## Key Features
- Implementation of four machine learning algorithms:
  - Logistic Regression (with Ridge, Lasso, and Elastic Net regularization)
  - K-Nearest Neighbors (K-NN)
  - Support Vector Machines (Linear and RBF kernels)
  - Random Forest
- Comprehensive data preprocessing pipeline
- Advanced model evaluation and comparison
- Handling of class imbalance
- Performance optimization through hyperparameter tuning

## Results
- Best performing model: Random Forest
  - 98.32% accuracy
  - 0.899 Kappa score
  - Lowest false-negative rate (26)
- Notable performances:
  - LASSO regression: 97.45% accuracy
  - SVM (Linear): 97.08% accuracy
  - K-NN: Highest specificity (93.32%)

## Dataset
HTRU2 Dataset features:
- 8 numerical features from pulsar candidates
- 17,898 total observations
- Class imbalance (~91% non-pulsars)

## Technical Implementation
- Data preprocessing:
  - Feature scaling
  - Class balancing
  - Parameter optimization
- Model evaluation:
  - 10-fold cross-validation
  - ROC curve analysis
  - Comprehensive metric comparison

## Technologies Used
- Python
- Scikit-learn
- Caret
- NumPy
- Pandas
- Matplotlib/Seaborn

## Professional Considerations
- Transparent methodology
- Reproducible results
- Ethical implications in astronomical research
- Scientific collaboration considerations

## Future Work
- Implementation of Convolutional Neural Networks
- Log transformations for linear models
- Integration with larger datasets
- Enhanced feature engineering

## Author
James Fothergill (v8255920)

  Click on the thumbnail to view the Demonstration video

[![YouTube Video Thumbnail](https://img.youtube.com/vi/Q2Xlj-bNVhc/0.jpg)](https://www.youtube.com/watch?v=Q2Xlj-bNVhc)
