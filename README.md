# Car Insurance Claim Outcome Prediction

A machine learning project that uses logistic regression to predict car insurance claim outcomes for "On the Road" insurance company. This project includes a complete web application built with Streamlit.

## 🚀 Features

- **Machine Learning Model**: Logistic regression model trained on synthetic insurance data
- **Web Application**: Interactive Streamlit app with modern UI
- **Feature Analysis**: Identifies key factors affecting claim probability
- **Data Visualization**: Interactive charts and insights
- **Real-time Predictions**: Predict claim likelihood based on customer information

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Generation

Generate the synthetic insurance claims dataset:
```bash
python generate_dataset.py
```

This creates `insurance_claims.csv` with 5,000 records containing:
- Customer demographics (age, gender)
- Driving history (experience, violations, previous claims)
- Vehicle information (type, age, annual mileage)
- Insurance details (coverage type, deductible)
- Credit score
- Target variable: claim_filed (0 or 1)

## 🤖 Model Training

Train the logistic regression model:
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train a logistic regression model
- Evaluate model performance
- Display feature importance
- Save the model and preprocessors to the `models/` directory

## 🌐 Running the Web Application

Launch the Streamlit web app:
```bash
streamlit run app.py
```

The application includes seven pages:

### 1. **🏠 Home**
- Welcome page with company overview
- Key features and benefits
- Quick statistics dashboard
- Call-to-action sections

### 2. **🔐 Login**
- User authentication interface
- Session management
- Demo mode (any username/password with 3+ characters works)

### 3. **📊 Predict Claim**
- Input customer information
- Get real-time claim probability prediction
- Interactive gauge visualization
- View risk factors and recommendations

### 4. **📈 Data Analysis**
- Overview statistics
- Interactive visualizations
- Feature correlations
- Claim distribution analysis

### 5. **🧠 Model Insights**
- Feature importance rankings
- Coefficient analysis
- Model performance metrics

### 6. **ℹ️ About**
- Company mission and vision
- Technology stack information
- Team overview
- Project details

### 7. **📞 Contact Us**
- Contact form with validation
- Office address and contact information
- Business hours
- Social media links

## 📈 Key Features Identified

The model identifies the following as key predictors:
- Previous claims history
- Traffic violations
- Vehicle type (Sports cars = higher risk)
- Credit score
- Annual mileage
- Age (younger drivers = higher risk)
- Deductible amount

## 🎯 Model Performance

The logistic regression model typically achieves:
- Accuracy: ~75-80%
- ROC-AUC Score: ~0.75-0.80

## 📁 Project Structure

```
DA project/
├── app.py                 # Streamlit web application
├── generate_dataset.py    # Dataset generation script
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── insurance_claims.csv  # Generated dataset (after running generate_dataset.py)
└── models/               # Saved models (after running train_model.py)
    ├── insurance_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    └── feature_cols.pkl
```

## 🔧 Usage Workflow

1. **Generate Dataset**: `python generate_dataset.py`
2. **Train Model**: `python train_model.py`
3. **Run App**: `streamlit run app.py`

## 💡 Business Applications

This model can be used for:
- Risk assessment and pricing
- Customer segmentation
- Underwriting decisions
- Premium calculation
- Fraud detection support

## 📝 Notes

- The dataset is synthetic and generated for demonstration purposes
- Real-world models would require actual insurance data
- Additional features and model tuning could improve performance
- Consider ensemble methods for production use

## 🤝 Contributing

Feel free to extend this project with:
- Additional machine learning models
- More sophisticated feature engineering
- Database integration
- API endpoints
- Advanced visualizations

## 📄 License

This project is for educational and demonstration purposes.

