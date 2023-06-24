# ArithmeticFeatures
A feature engineering tool based on simple arithmetic operations between each pair of numeric features. This is intended both to increase the accuracy and interpretability of models for some datasets. Experimental results are included below demonstrating its utility for some models on many datasets. 

The tool uses the same signature, based on the fit-tranform pattern, as sklearn's PolynomialFeatures and [RotationFeatures](https://github.com/Brett-Kennedy/RotationFeatures).

The tool simply generates additional numeric features through the application of basic arithmetic operations (+, -, *, /, and optionally min and max) to each pair of numeric features. It is posible to apply repeatedly, optionally interspersed  with feature selection, to create higher-order generated features, which may capture more complex feature interactions. In our experiments, executing once is typically sufficient to capture most feature interactions, and feature selection is often not necessary depending on the model using the generated features, though should usually be done to reduce overfitting and execution times. 

# Example

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

arith = ArithmeticFeatures()
extended_X = pd.DataFrame(arith.fit_transform(X), columns=arith.get_feature_names())
X_train, X_test, y_train, y_test = train_test_split(extended_X, y, random_state=42)

dt = tree.DecisionTreeClassifier(max_depth=4, random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
```

# Example Notebook

[Simple_Test_Arithmetic-Based_Feature_Generation](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/examples/Simple_Test_Arithmetic-Based_Feature_Generation.ipynb) Provides simple examples using the tool

# Accuracy Testing
[Accuracy_Test_ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/examples/Accuracy_Test_ArithmeticFeatures.py) Provides more thorough testing of the tool. This utilizes the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool, which can simplify testing on large numbers of datasets. The file compares both classification and regression problems on several sklearn predictors (Decision Trees, RandomForest, kNN, Logistion Regression, Lasso Linear Regression, Gaussian Naive Bayes, ExtraTrees, AdaBoost, and GradientBoost). Further, it performs tests using cross-validated grid search to determine the best settings for feature generation using ArithmeticFeatures. Some results are included below.

# Results
### Decision Trees
![Decision Trees](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/Results/results_26_08_2021_21_15_03_plot.png)
This shows two plots: the top for accuracy (higher is better) using a macro f1 score, and the second for complexity (smaller is better). It can be seen that while ArithmeticFeatures often provides for higher accuracy, the overal accuracy is quite similar. However, the model complexity (measured by number of nodes) is consistently lower, allowing for more interpretable models. The x-axis orders the datasets from least to highest accuracy on the baseline, the standard sklearn model with no generated features. 

### Random Forest
![Random Forest](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/Results/results_26_08_2021_21_25_45_plot.png)
This again shows the accuracy, though not consitently, often higher using ArithmeticFeatures.

### Logistic Regression
![Logistic Regression](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/Results/results_27_08_2021_10_41_11_plot.png)
Similar results as other models.

### Linear Discriminant Analysis
![Linear Discriminant Analysis](https://github.com/Brett-Kennedy/ArithmeticFeatures/blob/main/Results/results_27_08_2021_00_59_56_plot.png)
This shows similar results, but was slow to execute over 100 datasets, so was removed from the test file, along with QDA.

