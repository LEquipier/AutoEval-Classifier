# Car Evaluation Classifier — Multi-Model Comparison

## 1. Introduction

This project explores the Car Evaluation dataset to classify vehicle acceptability. **7 machine learning models** were trained on the training set and evaluated on the test set to compare their performance.

## 2. Dataset

The dataset comes from the **Car Evaluation Database**, originally derived from a simple hierarchical decision model developed for the DEX expert system (M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990).

The model evaluates cars according to the following concept structure:

```
CAR Evaluation              car acceptability
├── PRICE                   overall price
│   ├── buying              buying price
│   └── maint               maintenance price
└── TECH                    technical characteristics
    ├── COMFORT
    │   ├── doors           number of doors
    │   ├── persons         capacity in terms of persons to carry
    │   └── lug_boot        size of luggage boot
    └── safety              estimated safety of the car
```

The dataset removes intermediate structural information and directly relates the **6 input attributes** (buying, maint, doors, persons, lug_boot, safety) to the target class (CAR).

## 3. Models & Results

| Model | Accuracy |
|-------|----------|
| KNN | 94.13% |
| Naïve Bayes | 74.34% |
| Perceptron | 68.05% |
| Logistic Regression | 73.68% |
| Random Forest | **97.42%** |
| Linear SVC | 74.69% |
| Decision Tree | 97.24% |

### (a) KNN (K-Nearest Neighbors)

- **Preprocessing:** Read the data, transcoded the DataFrame using a mapping function, converted it to dictionary form, and split training/test sets at a 7:3 ratio.
- **Modeling:** Added feature labels and sorted in ascending order. Determined the optimal K value through testing and applied weighting to the "evaluation" results.
- **Accuracy: 94.13%** — Also computed running time and appended predictions to the test set.

### (b) Naïve Bayes

- **Preprocessing:** Used sklearn's `LabelEncoder()` to convert all string data to integers. The first 800 samples form the training set; the rest are the test set.
- **Modeling:** Implemented Bayesian classification by calling the `NaiveBayesClassifier()` algorithm.
- **Accuracy: 74.34%**

### (c) Perceptron

- **Preprocessing:** Encapsulated data preprocessing into a reusable function.
- **Modeling:** Set the learning rate and converted the three "evaluation" parameters into a 3D matrix. Randomly initialized W and b, then updated them via accumulated partial derivatives during training.
- **Accuracy: 68.05%** — The suboptimal result is likely due to inappropriate initial values of W and b.

### (d) Logistic Regression

- **Preprocessing:** Read and transcoded the data, then used sklearn's `train_test_split()` to split training and test sets.
- **Modeling:** Trained with sklearn's built-in model. After observing low initial accuracy and a learning curve showing accuracy decreasing with more data, `GridSearch` was used to tune the regularization parameter.
- **Accuracy: 73.68%** — As an imbalanced classification problem, accuracy alone is not the ideal evaluation metric.

### (e) Random Forest ⭐ Best Model

- **Preprocessing:** Used sklearn's `train_test_split()` to split training and test sets.
- **Modeling:** Used sklearn's built-in Random Forest algorithm with an initial accuracy of ~95%. Further tuning steps:
  1. Analyzed the effect of `n_estimators` — best performance at `n_estimators=30` (~96.3%); overfitting occurs beyond that.
  2. Analyzed the effect of `max_features` — best result at `max_features=5`.
  3. Used `GridSearch` to find the optimal parameter combination.
- **Final Accuracy: 97.42%** — The best-performing model in this project.

### (f) Linear SVC

- **Preprocessing:** Used sklearn's `train_test_split()` to split training and test sets.
- **Modeling:** Used sklearn's built-in Linear SVC algorithm.
- **Accuracy: 74.69%**

### (g) Decision Tree

- **Preprocessing:** Used sklearn's `train_test_split()` to split training and test sets.
- **Modeling:** Used sklearn's built-in Decision Tree algorithm.
- **Accuracy: 97.24%**

## 4. Project Files

| File | Description |
|------|-------------|
| `KNN.ipynb` | KNN model |
| `Naïve Bayes.ipynb` | Naïve Bayes model |
| `Perceptron.ipynb` | Perceptron model |
| `Logistic Regression.ipynb` | Logistic Regression model |
| `Random_Forest.ipynb` | Random Forest model |
| `Linear SVC.ipynb` | Linear SVC model |
| `Decision Tree.ipynb` | Decision Tree model |
| `training.csv` | Training dataset |
| `test.csv` | Test dataset |
| `test_RandomForest.csv` | Random Forest test results |

## 5. Conclusion

Among the 7 models, **Random Forest (97.42%)** and **Decision Tree (97.24%)** achieved the best performance, followed by KNN (94.13%). Linear models (Logistic Regression, Linear SVC, Naïve Bayes) and the Perceptron yielded relatively lower accuracy, indicating that the dataset's classification boundaries are highly nonlinear and better suited for ensemble learning or tree-based models.
