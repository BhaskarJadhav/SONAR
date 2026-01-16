# Sonar Data Classification (Rock vs. Mine)

## Project Overview
This project implements a binary classification model to distinguish between rocks and mines based on sonar data. The dataset consists of 60 numerical features representing the sonar signal bounced off an object, and a target label indicating whether the object is a 'Rock' (R) or a 'Mine' (M). A Logistic Regression model is trained and evaluated to perform this classification.

## How to Use

### Prerequisites
*   Python 3.x
*   Jupyter Notebook or Google Colab
*   Required Python libraries: `numpy`, `pandas`, `scikit-learn`

### Setup and Installation
1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/BhaskarJadhav/SONAR.git
    cd SONAR
    ```
2.  **Install Dependencies:**
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  **Download the Dataset:**
    The notebook assumes the dataset `Copy of sonar data.csv` is available. Please ensure this file is in the correct path or update the `pd.read_csv` line accordingly. A common source for this dataset is the UCI Machine Learning Repository (see 'Dataset Used' section below).
4.  **Run the Notebook:**
    Open and run the provided Jupyter Notebook (`.ipynb` file) or Google Colab notebook.

### Making Predictions
The notebook includes a prediction system section. To make a new prediction:
1.  Modify the `input_data` tuple in the prediction cell with your desired sonar readings.
2.  Run the prediction cells to get the classification result.

## How It Works

1.  **Data Collection and Processing:**
    *   The `sonar_data` is loaded from a CSV file into a Pandas DataFrame.
    *   Basic exploratory data analysis is performed (`.head()`, `.shape()`, `.describe()`, `.value_counts()`, `.groupby().mean()`) to understand the dataset structure and statistical properties.

2.  **Data Separation:**
    *   Features (sonar readings) are separated into `x` (all columns except the last one).
    *   Labels (Rock/Mine) are separated into `y` (the last column).

3.  **Train-Test Split:**
    *   The dataset is split into training and testing sets using `train_test_split` with a test size of 10% and `stratify=y` to maintain the class distribution.

4.  **Model Training:**
    *   A Logistic Regression model is initialized and trained (`model.fit()`) using the training data (`x_train`, `y_train`).

5.  **Model Evaluation:**
    *   The model's accuracy is calculated on both the training data (`x_train_prediction`, `y_train`) and the test data (`x_test_prediction`, `y_test`) using `accuracy_score`.

6.  **Prediction System:**
    *   A sample `input_data` (a tuple of 60 sonar readings) is converted into a NumPy array.
    *   The array is reshaped to `(1, -1)` for single-instance prediction.
    *   The trained model predicts the label ('R' for Rock or 'M' for Mine) for the input data.

## Tech Stack
*   **Python**
*   **NumPy**: For numerical operations, especially array manipulation.
*   **Pandas**: For data loading, manipulation, and analysis.
*   **Scikit-learn**: For machine learning tasks, including `LogisticRegression`, `train_test_split`, and `accuracy_score`.

## Expected Output and Application

### Expected Output
The model will output either:
*   `'R'` (indicating the object is a Rock) or
*   `'M'` (indicating the object is a Mine)

Along with a descriptive message like "The object is rock !!!" or "The object is mine !!".

### Application
This type of model can be applied in various real-world scenarios, such as:
*   **Underwater Object Detection**: Identifying potentially dangerous objects (like mines) from harmless ones (like rocks) in marine environments using sonar.
*   **Autonomous Underwater Vehicles (AUVs)**: Guiding AUVs to navigate and identify objects without human intervention.
*   **Geological Surveys**: Assisting in mapping underwater terrains and identifying different types of seabed compositions.

## Dataset Used

The dataset used in this project is the **Sonar, Mines vs. Rocks Dataset**.

*   **Description**: This dataset contains 111 patterns obtained by bouncing sonar signals off a metal cylinder (mine) at various angles and 97 patterns obtained from rocks at various angles. Each pattern is a set of 60 numbers in the range 0.0 to 1.0, representing the energy within a particular frequency band.
*   **Source**: You can typically find this dataset at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) or similar data science platforms like Kaggle.

*Please ensure the dataset file (`Copy of sonar data.csv`) is correctly placed in your working directory or its path is updated in the code.*
