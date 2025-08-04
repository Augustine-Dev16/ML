# Handwritten Digit Classification using SVM

This project demonstrates the use of a Support Vector Machine (SVM) classifier to recognize handwritten digits from images using a subset of the MNIST dataset provided by `sklearn.datasets`.

## Dataset

We use the **Digits Dataset** from `scikit-learn`, which contains 1,797 8x8 grayscale images of digits (0 through 9).

- Each image is 8x8 pixels
- Each pixel has an integer value from 0 to 16 representing the brightness
- Each image is flattened into a 64-length feature vector

## Objective

Train an SVM model to classify digits with high accuracy and visualize the predictions.

## Workflow

### 1. Data Loading and Visualization
- Load the digits dataset using `sklearn.datasets.load_digits()`
- Visualize a few sample images with their labels

### 2. Preprocessing
- Split the dataset into training and test sets (80/20 split)

### 3. Model Training
- Use `SVC` (Support Vector Classification) from `sklearn.svm`
- Kernel: RBF (Radial Basis Function)
- Hyperparameters:
  - `gamma = 0.001`
  - `C = 10`

### 4. Evaluation
- Print classification report
- Print overall accuracy
- Visualize predicted digits vs. actual labels

## Results

Sample accuracy: ~98% on test data  
(Exact performance may vary depending on the train-test split)

The model performs well even with minimal preprocessing due to the simplicity and clarity of the dataset.

## Requirements

Install the necessary libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## How to Run

Run the Python script in any environment that supports `matplotlib`:

```bash
python svm_digit_classifier.py
```

Or use a Jupyter Notebook for interactive execution and inline visualizations.

## File Structure

```
.
├── svm_digit_classifier.py
├── README.md
```

## License

This project is released for educational use under the MIT License.
