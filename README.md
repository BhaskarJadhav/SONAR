# Sonar Rock and Mine Classification

A Jupyter Notebook that uses logistic regression to classify sonar observations as rocks or mines.

## Workflow

- Load and inspect sonar measurements
- Separate 60 numerical features from the class label
- Create stratified training and test sets
- Train a logistic regression classifier
- Compare training and test accuracy
- Predict the class of a new sonar observation

## Dataset

The notebook expects the Sonar, Mines vs. Rocks dataset as a local CSV file. Each row contains 60 sonar signal measurements and a final label:

- `R` for rock
- `M` for mine

Update the CSV path in the notebook before running it.

## Run

```bash
pip install numpy pandas scikit-learn jupyter
jupyter notebook PROJECT1_SONAR_.ipynb
```

Run the notebook cells in order.

## Built with

`Python` `pandas` `NumPy` `scikit-learn`
