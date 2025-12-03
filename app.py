"""
Streamlit web application for Car Insurance Claim Prediction
On the Road Insurance - Claim Outcome Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="On the Road Insurance - Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .feature-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .contact-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_data
def load_model():
    try:
        model = joblib.load('models/insurance_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, scaler, label_encoders, feature_cols
    except FileNotFoundError:
        return None, None, None, None

model, scaler, label_encoders, feature_cols = load_model()

# Header
st.markdown('<h1 class="main-header">🚗 On the Road Insurance</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Car Insurance Claim Outcome Predictor</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")

# Show login status
if st.session_state.logged_in:
    st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.rerun()

# Navigation menu
menu_options = ["🏠 Home", "🔐 Login", "📊 Predict Claim", "📈 Data Analysis", "🧠 Model Insights", "ℹ️ About", "📞 Contact Us"]
page = st.sidebar.radio("Choose a page", menu_options)

if page == "🏠 Home":
    st.header("Welcome to On the Road Insurance")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚗 Your Trusted Partner in Car Insurance
        
        Welcome to **On the Road Insurance**, where cutting-edge machine learning meets 
        comprehensive car insurance solutions. Our advanced predictive analytics system 
        helps you understand and manage insurance risks with precision and confidence.
        """)
        
        st.markdown("""
        ### ✨ What We Offer
        
        - **AI-Powered Risk Assessment**: Predict claim probabilities using advanced logistic regression models
        - **Data-Driven Insights**: Comprehensive analysis of insurance trends and patterns
        - **Real-Time Predictions**: Get instant claim probability assessments
        - **Feature Analysis**: Understand key factors affecting insurance claims
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/1f77b4/ffffff?text=Insurance+AI", use_container_width=True)
    
    st.markdown("---")
    
    # Features section
    st.subheader("🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>🔮 Predictive Analytics</h4>
        <p>Advanced machine learning models to predict claim outcomes with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>📊 Data Visualization</h4>
        <p>Interactive charts and graphs to explore insurance data patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
        <h4>🧠 Model Insights</h4>
        <p>Understand which factors most influence claim probabilities</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    try:
        df = pd.read_csv('insurance_claims.csv')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Claim Rate", f"{df['claim_filed'].mean():.1%}")
        with col3:
            st.metric("Avg Age", f"{df['age'].mean():.0f} years")
        with col4:
            st.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
    except:
        st.info("Run 'generate_dataset.py' to see statistics")
    
    st.markdown("---")
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h2>Ready to Get Started?</h2>
    <p style="font-size: 1.2rem;">Navigate to <strong>Predict Claim</strong> to start making predictions!</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔐 Login":
    st.header("🔐 User Login")
    st.markdown("---")
    
    if st.session_state.logged_in:
        st.success(f"✅ You are already logged in as {st.session_state.username}")
        st.info("Navigate to other pages to access all features.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
            """, unsafe_allow_html=True)
            
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Login", type="primary", use_container_width=True):
                    # Simple authentication (for demo purposes)
                    # In production, use proper authentication with database
                    if username and password:
                        if len(username) >= 3 and len(password) >= 3:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success(f"✅ Login successful! Welcome, {username}!")
                            st.rerun()
                        else:
                            st.error("❌ Username and password must be at least 3 characters")
                    else:
                        st.error("❌ Please enter both username and password")
            
            with col_btn2:
                if st.button("Register", use_container_width=True):
                    st.info("Registration feature coming soon!")
            
            st.markdown("""
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.info("💡 **Demo Mode**: For demonstration purposes, any username/password with at least 3 characters will work.")

elif page == "📊 Predict Claim":
    if model is None:
        st.error("Model files not found. Please run 'train_model.py' first.")
    else:
        st.header("📊 Claim Prediction")
        st.markdown("Enter customer information to predict the likelihood of filing a claim.")
        
        if not st.session_state.logged_in:
            st.info("💡 You can use this feature without logging in, but logging in provides a better experience!")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            driving_experience = st.slider("Driving Experience (years)", 0, 50, 10)
            vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
            vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Sports Car", "Truck", "Hatchback"])
            annual_mileage = st.slider("Annual Mileage", 5000, 30000, 15000)
        
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            previous_claims = st.number_input("Previous Claims", 0, 10, 0, step=1)
            traffic_violations = st.number_input("Traffic Violations", 0, 10, 0, step=1)
            coverage_type = st.selectbox("Coverage Type", ["Basic", "Standard", "Premium"])
            deductible = st.selectbox("Deductible ($)", [500, 1000, 2000, 5000])
        
        # Predict button
        if st.button("🔮 Predict Claim Probability", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'age': age,
                'gender': gender,
                'driving_experience': driving_experience,
                'vehicle_age': vehicle_age,
                'vehicle_type': vehicle_type,
                'annual_mileage': annual_mileage,
                'credit_score': credit_score,
                'previous_claims': previous_claims,
                'traffic_violations': traffic_violations,
                'coverage_type': coverage_type,
                'deductible': deductible
            }
            
            # Encode categorical variables
            input_encoded = {}
            for col in feature_cols:
                if col in label_encoders:
                    input_encoded[col] = label_encoders[col].transform([input_data[col]])[0]
                else:
                    input_encoded[col] = input_data[col]
            
            # Create feature array
            X_input = pd.DataFrame([input_encoded], columns=feature_cols)
            X_input_scaled = scaler.transform(X_input)
            
            # Make prediction
            probability = model.predict_proba(X_input_scaled)[0][1]
            prediction = model.predict(X_input_scaled)[0]
            
            # Display results
            st.markdown("---")
            
            # Probability visualization
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Claim Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Result boxes
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Outcome", "Will File Claim" if prediction == 1 else "No Claim")
            with col2:
                st.metric("Confidence", f"{probability:.1%}")
            
            # Risk factors
            st.subheader("🔍 Key Risk Factors")
            risk_factors = []
            if age < 30:
                risk_factors.append("Young driver (higher risk)")
            if vehicle_type == "Sports Car":
                risk_factors.append("Sports car (higher risk)")
            if annual_mileage > 20000:
                risk_factors.append("High annual mileage")
            if credit_score < 600:
                risk_factors.append("Low credit score")
            if previous_claims > 0:
                risk_factors.append(f"{previous_claims} previous claim(s)")
            if traffic_violations > 0:
                risk_factors.append(f"{traffic_violations} traffic violation(s)")
            if deductible == 500:
                risk_factors.append("Low deductible (more likely to file)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("✅ Low risk profile")

elif page == "📈 Data Analysis":
    st.header("📈 Data Analysis")
    
    try:
        df = pd.read_csv('insurance_claims.csv')
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Claim Rate", f"{df['claim_filed'].mean():.1%}")
        with col3:
            st.metric("Average Age", f"{df['age'].mean():.0f}")
        with col4:
            st.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Claim distribution by vehicle type
            fig = px.bar(
                df.groupby('vehicle_type')['claim_filed'].mean().reset_index(),
                x='vehicle_type',
                y='claim_filed',
                title='Claim Rate by Vehicle Type',
                labels={'claim_filed': 'Claim Rate', 'vehicle_type': 'Vehicle Type'},
                color='claim_filed',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Age distribution
            fig = px.histogram(
                df, x='age', color='claim_filed',
                title='Age Distribution by Claim Status',
                labels={'age': 'Age', 'claim_filed': 'Claim Filed'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Claim rate by coverage type
            fig = px.pie(
                df.groupby(['coverage_type', 'claim_filed']).size().reset_index(name='count'),
                values='count',
                names='coverage_type',
                title='Distribution by Coverage Type',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Credit score vs claims
            fig = px.box(
                df, x='claim_filed', y='credit_score',
                title='Credit Score Distribution by Claim Status',
                labels={'claim_filed': 'Claim Filed', 'credit_score': 'Credit Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please run 'generate_dataset.py' first.")

elif page == "🧠 Model Insights":
    st.header("🧠 Model Insights")
    
    try:
        # Load feature importance
        df = pd.read_csv('insurance_claims.csv')
        model_coef = model.coef_[0]
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model_coef,
            'Importance': np.abs(model_coef)
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        st.markdown("The following features have the strongest impact on claim predictions:")
        
        # Feature importance chart
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Absolute Coefficient Values)',
            color='Coefficient',
            color_continuous_scale='RdBu',
            labels={'Importance': 'Absolute Coefficient Value'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed feature analysis
        st.subheader("Feature Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Impact (Increases Claim Probability):**")
            positive_features = feature_importance_df[feature_importance_df['Coefficient'] > 0].sort_values('Coefficient', ascending=False)
            for _, row in positive_features.iterrows():
                st.write(f"• **{row['Feature']}**: {row['Coefficient']:.4f}")
        
        with col2:
            st.markdown("**Negative Impact (Decreases Claim Probability):**")
            negative_features = feature_importance_df[feature_importance_df['Coefficient'] < 0].sort_values('Coefficient')
            for _, row in negative_features.iterrows():
                st.write(f"• **{row['Feature']}**: {row['Coefficient']:.4f}")
        
        # Model performance metrics
        st.subheader("Model Performance")
        st.info("""
        **Logistic Regression Model**
        - This model uses logistic regression to predict binary claim outcomes
        - Features are standardized before training
        - The model provides probability scores for claim likelihood
        - Key features identified through coefficient analysis
        """)
        
    except Exception as e:
        st.error(f"Error loading model insights: {str(e)}")

elif page == "ℹ️ About":
    st.header("ℹ️ About On the Road Insurance")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Our Mission
        
        At **On the Road Insurance**, we combine the power of artificial intelligence 
        with decades of insurance expertise to provide accurate, data-driven risk 
        assessments. Our mission is to revolutionize the insurance industry through 
        innovative machine learning solutions.
        
        ### Our Technology
        
        This application uses **Logistic Regression**, a powerful machine learning 
        algorithm that predicts binary outcomes (claim filed or not filed) based on 
        various customer and vehicle characteristics. Our model has been trained on 
        thousands of insurance records to identify key risk factors.
        
        ### Key Features
        
        - **Predictive Modeling**: Advanced logistic regression for claim prediction
        - **Feature Analysis**: Identifies the most important factors affecting claims
        - **Data Visualization**: Interactive charts and graphs for data exploration
        - **Real-Time Predictions**: Instant claim probability assessments
        - **Risk Assessment**: Comprehensive analysis of customer risk profiles
        
        ### Model Performance
        
        Our logistic regression model achieves:
        - **Accuracy**: ~75-80%
        - **ROC-AUC Score**: ~0.75-0.80
        - **Key Predictors**: Previous claims, traffic violations, vehicle type, 
          credit score, annual mileage, age, and deductible amount
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Project Details
        
        **Project Type**: Machine Learning Application
        
        **Technology Stack**:
        - Python 3.8+
        - Streamlit (Web Framework)
        - Scikit-learn (ML Library)
        - Pandas & NumPy (Data Processing)
        - Plotly (Visualization)
        
        **Model Type**: Logistic Regression
        
        **Dataset**: Synthetic insurance data (5,000 records)
        """)
    
    st.markdown("---")
    
    st.subheader("👥 Our Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Data Science Team**
        - Machine Learning Engineers
        - Data Analysts
        - Risk Assessment Specialists
        """)
    with col2:
        st.markdown("""
        **Development Team**
        - Full-Stack Developers
        - UI/UX Designers
        - DevOps Engineers
        """)
    with col3:
        st.markdown("""
        **Business Team**
        - Insurance Experts
        - Actuaries
        - Customer Success
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Project Information")
    st.info("""
    This is a demonstration project showcasing the application of machine learning 
    in the insurance industry. The model uses logistic regression to predict car 
    insurance claim outcomes based on customer demographics, driving history, 
    vehicle information, and insurance details.
    
    **Note**: This application uses synthetic data for demonstration purposes. 
    In a production environment, real insurance data would be used with proper 
    security and privacy measures.
    """)

elif page == "📞 Contact Us":
    st.header("📞 Contact Us")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Get in Touch")
        st.markdown("""
        We'd love to hear from you! Whether you have questions about our services, 
        need technical support, or want to learn more about our machine learning 
        solutions, we're here to help.
        """)
        
        # Contact form
        with st.form("contact_form"):
            name = st.text_input("👤 Your Name *", placeholder="Enter your full name")
            email = st.text_input("📧 Email Address *", placeholder="your.email@example.com")
            phone = st.text_input("📱 Phone Number", placeholder="+1 (555) 123-4567")
            subject = st.selectbox("📋 Subject", [
                "General Inquiry",
                "Technical Support",
                "Business Partnership",
                "Feature Request",
                "Other"
            ])
            message = st.text_area("💬 Message *", placeholder="Enter your message here...", height=150)
            
            submitted = st.form_submit_button("Send Message", type="primary", use_container_width=True)
            
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you for your message! We'll get back to you soon.")
                    st.balloons()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")
    
    with col2:
        st.subheader("Contact Information")
        
        st.markdown("""
        <div class="contact-card">
        <h4>📍 Office Address</h4>
        <p>123 Insurance Boulevard<br>
        Suite 500<br>
        New York, NY 10001<br>
        United States</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
        <h4>📞 Phone</h4>
        <p>Main: +1 (555) 123-4567<br>
        Support: +1 (555) 123-HELP<br>
        Fax: +1 (555) 123-4568</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
        <h4>📧 Email</h4>
        <p>General: info@ontheroadinsurance.com<br>
        Support: support@ontheroadinsurance.com<br>
        Sales: sales@ontheroadinsurance.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
        <h4>🕒 Business Hours</h4>
        <p>Monday - Friday: 9:00 AM - 6:00 PM EST<br>
        Saturday: 10:00 AM - 4:00 PM EST<br>
        Sunday: Closed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Social media and additional info
    st.subheader("🌐 Connect With Us")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**LinkedIn**\n\n[Connect with us](https://linkedin.com)")
    with col2:
        st.markdown("**Twitter**\n\n[Follow us](https://twitter.com)")
    with col3:
        st.markdown("**Facebook**\n\n[Like our page](https://facebook.com)")
    with col4:
        st.markdown("**GitHub**\n\n[View our code](https://github.com)")
    
    st.markdown("---")
    
    # Map placeholder
    st.subheader("📍 Find Us")
    st.info("🗺️ Interactive map would be displayed here in a production environment")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "On the Road Insurance - Machine Learning Powered Claim Prediction System | "
    "© 2024 All Rights Reserved"
    "</div>",
    unsafe_allow_html=True
)

