AgriNutri AI – Smart Farming and Nutrition Recommendation System

Overview

AgriNutri AI is a machine learning-based smart farming application designed to support farmers in making informed agricultural decisions. The system analyzes important soil and environmental parameters to recommend suitable crops and provide agriculture-related nutrition recommendations.

It uses a Random Forest machine learning algorithm to analyze input parameters and generate crop recommendations.

Features

- Crop recommendation based on soil and environmental conditions.
- Uses essential soil nutrients such as Nitrogen (N), Phosphorus (P), and Potassium (K).
- Considers temperature, humidity, soil pH, and rainfall.
- Provides data-driven recommendations using machine learning.
- Includes dataset preparation and model evaluation scripts.
- User-friendly web interface.

Technologies Used

- Programming Language: Python
- Backend: Flask
- Machine Learning: Scikit-learn
- Data Processing: Pandas, NumPy
- Algorithm: Random Forest Classifier
- Frontend: HTML, CSS

Input Parameters

The crop recommendation model uses seven important parameters:

1. Nitrogen (N)
2. Phosphorus (P)
3. Potassium (K)
4. Temperature
5. Humidity
6. Soil pH
7. Rainfall

How It Works

1. The user enters the required soil and environmental parameters.
2. The application processes the input data.
3. The trained Random Forest model analyzes the parameters.
4. The system predicts a suitable crop based on the input conditions.
5. The application displays the recommendation to the user.

Machine Learning Approach

The project uses the Random Forest algorithm, an ensemble learning technique that combines multiple decision trees to make predictions.

The model is trained using agricultural datasets and evaluated to understand its predictive performance.

Project Structure

AgriNutri-AI/
│
├── app.py
├── static/
├── check_crop_food_mapping.py
├── check_dataset.py
├── check_nutrition_dataset.py
├── compare_crop_models.py
├── evaluate_crop_model.py
├── prepare_nutrition_dataset.py
├── test_crop_model.py
├── confusion_matrix.png
└── model_comparison_results.csv

Installation and Setup

1. Clone the repository

git clone https://github.com/rutujamanerajure2005-hash/AgriNutri-AI.git

2. Navigate to the project folder

cd AgriNutri-AI

3. Install the required libraries

pip install flask pandas numpy scikit-learn

4. Run the application

python app.py

5. Open in your browser

http://127.0.0.1:5000/

Project Objective

The main objective of AgriNutri AI is to apply machine learning techniques to agricultural data and support smarter farming decisions through data-driven crop recommendations.

---

Developed as an academic project using Python, Flask and Machine Learning.
