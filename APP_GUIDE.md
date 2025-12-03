# 🚗 On the Road Insurance - Application Guide

## 🌐 Access Your Application

**URL:** http://localhost:8501

The application should automatically open in your default web browser. If not, manually navigate to the URL above.

---

## 📱 Application Overview

### 🏠 **HOME PAGE**
**What you'll see:**
- Large header: "🚗 On the Road Insurance"
- Welcome message and company overview
- Three feature boxes:
  - 🔮 Predictive Analytics
  - 📊 Data Visualization  
  - 🧠 Model Insights
- Quick Statistics dashboard showing:
  - Total Records: 5,000
  - Claim Rate: ~37-38%
  - Average Age
  - Average Credit Score
- Call-to-action section at the bottom

---

### 🔐 **LOGIN PAGE**
**What you'll see:**
- Login form with:
  - Username field
  - Password field (hidden)
  - "Login" button (primary/blue)
  - "Register" button
- Demo mode notice at the bottom
- **To login:** Enter any username and password with 3+ characters each
  - Example: username: `demo`, password: `demo123`
- After login, you'll see your username in the sidebar

---

### 📊 **PREDICT CLAIM PAGE**
**What you'll see:**
- Two-column input form:
  
  **Left Column:**
  - Age slider (18-80)
  - Gender dropdown (Male/Female)
  - Driving Experience slider (0-50 years)
  - Vehicle Age slider (0-20 years)
  - Vehicle Type dropdown (Sedan, SUV, Sports Car, Truck, Hatchback)
  - Annual Mileage slider (5,000-30,000)
  
  **Right Column:**
  - Credit Score slider (300-850)
  - Previous Claims number input (0-10)
  - Traffic Violations number input (0-10)
  - Coverage Type dropdown (Basic, Standard, Premium)
  - Deductible dropdown ($500, $1000, $2000, $5000)

- Large blue "🔮 Predict Claim Probability" button

**After clicking Predict:**
- Gauge chart showing claim probability (0-100%)
- Two metric boxes:
  - Predicted Outcome (Will File Claim / No Claim)
  - Confidence percentage
- Risk Factors section showing warnings or success messages

---

### 📈 **DATA ANALYSIS PAGE**
**What you'll see:**
- Four metric cards at the top:
  - Total Records
  - Claim Rate
  - Average Age
  - Average Credit Score

- **Visualizations (2 columns):**
  - **Left Column:**
    - Bar chart: Claim Rate by Vehicle Type
    - Histogram: Age Distribution by Claim Status
  
  - **Right Column:**
    - Pie chart: Distribution by Coverage Type
    - Box plot: Credit Score Distribution by Claim Status

- **Full-width:**
  - Correlation Matrix heatmap showing relationships between features

---

### 🧠 **MODEL INSIGHTS PAGE**
**What you'll see:**
- **Feature Importance Chart:**
  - Horizontal bar chart showing which features most affect predictions
  - Color-coded (red = increases risk, blue = decreases risk)

- **Feature Analysis (2 columns):**
  - **Left:** Positive Impact features (increase claim probability)
  - **Right:** Negative Impact features (decrease claim probability)

- **Model Performance section:**
  - Information box explaining the logistic regression model
  - Accuracy and ROC-AUC metrics

---

### ℹ️ **ABOUT PAGE**
**What you'll see:**
- **Our Mission** section
- **Our Technology** section explaining:
  - Logistic Regression model
  - Key features
  - Model performance
- **Project Details** sidebar with:
  - Technology stack
  - Model type
  - Dataset information
- **Our Team** section (3 columns)
- **Project Information** box

---

### 📞 **CONTACT US PAGE**
**What you'll see:**
- **Left Side:** Contact Form
  - Name field (required)
  - Email field (required)
  - Phone field (optional)
  - Subject dropdown
  - Message textarea (required)
  - "Send Message" button
  - Success/error messages after submission

- **Right Side:** Contact Information Cards
  - 📍 Office Address
  - 📞 Phone numbers
  - 📧 Email addresses
  - 🕒 Business Hours

- **Bottom:**
  - Social media links (LinkedIn, Twitter, Facebook, GitHub)
  - Map placeholder

---

## 🎨 Design Features

- **Color Scheme:**
  - Primary blue (#1f77b4) for headers
  - Gradient purple backgrounds for call-to-action sections
  - Clean white cards with subtle shadows
  - Color-coded visualizations

- **Layout:**
  - Wide layout for better visualization
  - Responsive columns
  - Modern card-based design
  - Professional typography

- **Interactive Elements:**
  - Sliders for numeric inputs
  - Dropdowns for categorical choices
  - Real-time predictions
  - Interactive Plotly charts

---

## 🚀 Quick Start Guide

1. **Open the app** → http://localhost:8501
2. **Explore Home** → Read about the application
3. **Try Login** → Use `demo` / `demo123`
4. **Make a Prediction** → Go to Predict Claim, adjust sliders, click Predict
5. **View Analysis** → Check Data Analysis for charts
6. **See Insights** → Review Model Insights for feature importance
7. **Learn More** → Read About page
8. **Contact** → Fill out Contact Us form

---

## 💡 Tips

- **Navigation:** Use the sidebar radio buttons to switch between pages
- **Login Status:** When logged in, your username appears in the sidebar
- **Predictions:** Try different combinations to see how risk factors change
- **Charts:** All charts are interactive - hover for details, zoom, pan
- **Forms:** Required fields are marked with asterisks (*)

---

## 🛑 Stopping the Server

To stop the Streamlit server:
- Press `Ctrl+C` in the terminal
- Or close the terminal window

---

Enjoy exploring the Car Insurance Claim Prediction System! 🚗📊

