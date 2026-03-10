# 🧠 Machine Learning Prediction API using Flask

This project demonstrates how to build a **Machine Learning API using Python and Flask**.
The trained model is exposed through an API endpoint so users can send input data and receive predictions.

---

# 📌 Project Overview

This repository contains:

1️⃣ **Classification Model**

* Dataset: Social Network Ads
* Algorithm: Decision Tree
* Task: Predict whether a user purchases a product.

2️⃣ **Regression Model**

* Dataset: Insurance
* Algorithm: Decision Tree Regressor
* Task: Predict insurance charges based on user attributes.

3️⃣ **Flask API**

* Loads the trained model
* Accepts input data through API
* Returns predictions in JSON format

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask
* Joblib

---

# 📂 Project Structure

```
project-folder
│
├── Social_Network_Ads - Social_Network_Ads.csv
├── insurance - insurance.csv
│
├── train_model.py
├── regression_model.pkl
├── dt_model.pkl
│
├── app.py
│
└── README.md
```

---

# ⚙️ Model Training

The model is trained using **Decision Tree Regressor**.

Steps:

1. Load dataset
2. Encode categorical variables
3. Split data into training and testing sets
4. Train Decision Tree model
5. Save the model using Joblib

Example:

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
```

The trained model is saved as:

```
regression_model.pkl
```

---

# 🚀 Running the Flask API

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn flask joblib
```

### 2️⃣ Run the API

```bash
```
