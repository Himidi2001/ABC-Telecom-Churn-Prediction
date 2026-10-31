# ABC Telecom Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for ABC Telecom using machine learning techniques. Customer churn occurs when customers discontinue a company's services. The project analyses customer data to identify patterns associated with churn and develops predictive models to estimate which customers are more likely to leave. Three machine learning algorithms are implemented and compared to identify the model with the best predictive performance.

## Dataset
The project uses the ABC Telecom customer dataset, which contains 3,333 customer records and 20 columns. The dataset includes customer account information, service usage details and churn status. The target variable is `Churn`, which indicates whether a customer has left the company.

## Technologies Used
- Python – Programming language
- Pandas – Data handling and analysis
- NumPy – Numerical operations
- Scikit-learn – Data preprocessing and machine learning
  
## Data Preprocessing
The dataset was checked for missing values and prepared before model training. Missing numerical values were handled using median imputation, while missing categorical values were handled using the most frequent value. Categorical features were converted into numerical format using One-Hot Encoding, and numerical features were standardised using StandardScaler. The dataset was divided into training and testing sets using an 80:20 split.

## Machine Learning Algorithms

Three machine learning algorithms were implemented and evaluated:

1. **Logistic Regression:** Used as a baseline classification model.
2. **Decision Tree:** Used to identify patterns and decision rules in customer data.
3. **Random Forest:** Used an ensemble of decision trees to improve prediction performance.

The models were compared using their accuracy and classification performance to identify the most suitable model for predicting customer churn.

## Model Performance

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 85.61% |
| Decision Tree | 90.10% |
| Random Forest | 93.40% |

## Confusion Matrix

| Actual / Predicted | Stayed | Churned |
|---|---:|---:|
| Stayed | 565 | 5 |
| Churned | 39 | 58 |

## How to Run the Project

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install the required Python libraries.
4. Run the Python file:

   ```bash
   python main.py
