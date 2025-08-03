"""
Experiment 4: Machine Learning Modeling & Performance Evaluation
Comprehensive ML pipeline with multiple algorithms and evaluation metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

class Experiment4:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        self.data_path = r"C:\Users\Omkar\SEM_7\DS_Project\datasets\DS-2-8-25"
        self.datasets = {}
        self.models = {}
        self.results = {}
        
    def render_ml_dashboard(self):
        """Main ML modeling dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>🤖 Machine Learning Modeling & Performance Evaluation</h2>
            <p>Comprehensive ML pipeline with algorithm comparison and performance analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load datasets
        self._load_datasets()
        
        if not self.datasets:
            st.error("❌ No datasets found. Please ensure data files are available.")
            return
        
        # ML Pipeline Options
        ml_options = [
            "🎯 Problem Definition & Data Preparation",
            "🔄 Data Preprocessing & Feature Engineering",
            "⚙️ Model Selection & Training",
            "📊 Model Performance Evaluation",
            "📈 Cross-Validation & Hyperparameter Tuning",
            "🔍 Feature Importance Analysis",
            "📋 Model Comparison Dashboard",
            "🚀 Prediction Interface",
            "💾 Model Export & Deployment Prep"
        ]
        
        selected_activity = st.selectbox(
            "Select ML Activity:",
            ml_options
        )
        
        # Route to appropriate activity
        if selected_activity == "🎯 Problem Definition & Data Preparation":
            self.problem_definition()
        elif selected_activity == "🔄 Data Preprocessing & Feature Engineering":
            self.data_preprocessing()
        elif selected_activity == "⚙️ Model Selection & Training":
            self.model_training()
        elif selected_activity == "📊 Model Performance Evaluation":
            self.model_evaluation()
        elif selected_activity == "📈 Cross-Validation & Hyperparameter Tuning":
            self.hyperparameter_tuning()
        elif selected_activity == "🔍 Feature Importance Analysis":
            self.feature_importance_analysis()
        elif selected_activity == "📋 Model Comparison Dashboard":
            self.model_comparison()
        elif selected_activity == "🚀 Prediction Interface":
            self.prediction_interface()
        elif selected_activity == "💾 Model Export & Deployment Prep":
            self.model_deployment_prep()

    def problem_definition(self):
        """Define ML problem and prepare target variables"""
        st.subheader("🎯 Problem Definition & Data Preparation")
        
        # Dataset selection
        dataset_names = list(self.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for ML Modeling:", dataset_names)
        df = self.datasets[selected_dataset]
        
        st.subheader("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Problem type selection
        st.subheader("🎯 Define ML Problem")
        
        problem_types = [
            "Classification - Predict Rating Categories",
            "Classification - Predict Sentiment",
            "Classification - Predict Product Category",
            "Regression - Predict Price",
            "Clustering - Customer Segmentation"
        ]
        
        selected_problem = st.selectbox("Select Problem Type:", problem_types)
        
        # Target variable definition based on problem type
        if "Rating Categories" in selected_problem:
            self._define_rating_classification(df)
        elif "Sentiment" in selected_problem:
            self._define_sentiment_classification(df)
        elif "Product Category" in selected_problem:
            self._define_category_classification(df)
        elif "Price" in selected_problem:
            self._define_price_regression(df)
        elif "Customer Segmentation" in selected_problem:
            self._define_clustering_problem(df)

    def _define_rating_classification(self, df):
        """Define rating classification problem"""
        st.subheader("⭐ Rating Classification Problem")
        
        if 'rating' not in df.columns:
            st.warning("⚠️ Rating column not found in dataset")
            return
        
        # Create rating categories
        st.write("**Target Variable Creation:**")
        df['rating_category'] = pd.cut(df['rating'], 
                                     bins=[0, 2, 3, 4, 5], 
                                     labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Display distribution
        rating_dist = df['rating_category'].value_counts()
        
        fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                    title="Rating Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Store in session state for later use
        if 'ml_data' not in st.session_state:
            st.session_state.ml_data = {}
        st.session_state.ml_data['target'] = 'rating_category'
        st.session_state.ml_data['problem_type'] = 'classification'
        st.session_state.ml_data['df'] = df

    def _define_sentiment_classification(self, df):
        """Define sentiment classification problem"""
        st.subheader("😊 Sentiment Classification Problem")
        
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        
        if not sentiment_cols:
            st.warning("⚠️ Sentiment column not found in dataset")
            return
        
        target_col = sentiment_cols[0]
        sentiment_dist = df[target_col].value_counts()
        
        fig = px.pie(values=sentiment_dist.values, names=sentiment_dist.index,
                    title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Store in session state
        if 'ml_data' not in st.session_state:
            st.session_state.ml_data = {}
        st.session_state.ml_data['target'] = target_col
        st.session_state.ml_data['problem_type'] = 'classification'
        st.session_state.ml_data['df'] = df

    def _define_category_classification(self, df):
        """Define category classification problem"""
        st.subheader("🏷️ Product Category Classification")
        
        if 'category' not in df.columns:
            st.warning("⚠️ Category column not found in dataset")
            return
        
        category_dist = df['category'].value_counts()
        
        fig = px.bar(x=category_dist.values, y=category_dist.index,
                    orientation='h',
                    title="Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Store in session state
        if 'ml_data' not in st.session_state:
            st.session_state.ml_data = {}
        st.session_state.ml_data['target'] = 'category'
        st.session_state.ml_data['problem_type'] = 'classification'
        st.session_state.ml_data['df'] = df

    def _define_price_regression(self, df):
        """Define price regression problem"""
        st.subheader("💰 Price Regression Problem")
        
        if 'price' not in df.columns:
            st.warning("⚠️ Price column not found in dataset")
            return
        
        # Price distribution
        fig = px.histogram(df, x='price', nbins=50,
                          title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Price", f"${df['price'].mean():.2f}")
        with col2:
            st.metric("Median Price", f"${df['price'].median():.2f}")
        with col3:
            st.metric("Min Price", f"${df['price'].min():.2f}")
        with col4:
            st.metric("Max Price", f"${df['price'].max():.2f}")
        
        # Store in session state
        if 'ml_data' not in st.session_state:
            st.session_state.ml_data = {}
        st.session_state.ml_data['target'] = 'price'
        st.session_state.ml_data['problem_type'] = 'regression'
        st.session_state.ml_data['df'] = df

    def _define_clustering_problem(self, df):
        """Define clustering problem"""
        st.subheader("👥 Customer Segmentation Problem")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            st.warning("⚠️ Need at least 2 numerical columns for clustering")
            return
        
        selected_features = st.multiselect("Select Features for Clustering:", numerical_cols)
        
        if len(selected_features) >= 2:
            # Show feature correlation
            corr_matrix = df[selected_features].corr()
            
            fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Store in session state
            if 'ml_data' not in st.session_state:
                st.session_state.ml_data = {}
            st.session_state.ml_data['features'] = selected_features
            st.session_state.ml_data['problem_type'] = 'clustering'
            st.session_state.ml_data['df'] = df

    def data_preprocessing(self):
        """Data preprocessing and feature engineering"""
        st.subheader("🔄 Data Preprocessing & Feature Engineering")
        
        if 'ml_data' not in st.session_state:
            st.warning("⚠️ Please define the ML problem first")
            return
        
        df = st.session_state.ml_data['df'].copy()
        
        # Feature selection
        st.subheader("🎯 Feature Selection")
        
        all_columns = df.columns.tolist()
        if 'target' in st.session_state.ml_data:
            target_col = st.session_state.ml_data['target']
            feature_columns = [col for col in all_columns if col != target_col]
        else:
            feature_columns = all_columns
        
        selected_features = st.multiselect(
            "Select Features for Model:",
            feature_columns,
            default=feature_columns[:min(10, len(feature_columns))]
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature")
            return
        
        # Data cleaning options
        st.subheader("🧹 Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values:",
                ["Drop rows with missing values", "Fill with mean/mode", "Forward fill"]
            )
        
        with col2:
            remove_outliers = st.checkbox("Remove outliers (IQR method)")
        
        # Apply preprocessing
        processed_df = df[selected_features + ([st.session_state.ml_data['target']] if 'target' in st.session_state.ml_data else [])].copy()
        
        # Handle missing values
        if handle_missing == "Drop rows with missing values":
            processed_df = processed_df.dropna()
        elif handle_missing == "Fill with mean/mode":
            for col in processed_df.columns:
                if processed_df[col].dtype in ['int64', 'float64']:
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                else:
                    processed_df[col].fillna(processed_df[col].mode().iloc[0] if len(processed_df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove outliers
        if remove_outliers:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
        
        # Encoding categorical variables
        st.subheader("🔄 Categorical Encoding")
        
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'target' in st.session_state.ml_data and st.session_state.ml_data['target'] in categorical_cols:
            categorical_cols.remove(st.session_state.ml_data['target'])
        
        if categorical_cols:
            encoding_method = st.selectbox(
                "Select Encoding Method:",
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            if encoding_method == "Label Encoding":
                for col in categorical_cols:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            else:  # One-Hot Encoding
                processed_df = pd.get_dummies(processed_df, columns=categorical_cols)
        
        # Feature scaling
        st.subheader("⚖️ Feature Scaling")
        
        scale_features = st.checkbox("Apply Standard Scaling to numerical features")
        
        if scale_features:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in st.session_state.ml_data and st.session_state.ml_data['target'] in numerical_cols:
                numerical_cols.remove(st.session_state.ml_data['target'])
            
            scaler = StandardScaler()
            processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
        
        # Display preprocessing results
        st.subheader("📊 Preprocessing Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Records", len(df))
        with col2:
            st.metric("Processed Records", len(processed_df))
        with col3:
            st.metric("Final Features", len(processed_df.columns) - (1 if 'target' in st.session_state.ml_data else 0))
        
        # Store processed data
        st.session_state.ml_data['processed_df'] = processed_df
        
        # Show sample of processed data
        st.subheader("🔍 Processed Data Sample")
        st.dataframe(processed_df.head(), use_container_width=True)

    def model_training(self):
        """Train multiple ML models"""
        st.subheader("⚙️ Model Selection & Training")
        
        if 'ml_data' not in st.session_state or 'processed_df' not in st.session_state.ml_data:
            st.warning("⚠️ Please complete data preprocessing first")
            return
        
        df = st.session_state.ml_data['processed_df']
        problem_type = st.session_state.ml_data['problem_type']
        
        if problem_type == 'clustering':
            self._clustering_analysis(df)
            return
        
        target_col = st.session_state.ml_data['target']
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle target encoding for classification
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.ml_data['label_encoder'] = le
        
        # Train-test split
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        
        if problem_type == 'classification':
            available_models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "SVM": SVC(random_state=42, probability=True),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Naive Bayes": GaussianNB()
            }
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.svm import SVR
            
            available_models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "SVR": SVR()
            }
        
        selected_models = st.multiselect(
            "Select Models to Train:",
            list(available_models.keys()),
            default=list(available_models.keys())[:3]
        )
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
            return
        
        # Train models
        if st.button("🚀 Train Models"):
            trained_models = {}
            training_results = {}
            
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(selected_models):
                st.write(f"Training {model_name}...")
                
                model = available_models[model_name]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store results
                trained_models[model_name] = model
                training_results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            # Store in session state
            st.session_state.ml_data['trained_models'] = trained_models
            st.session_state.ml_data['training_results'] = training_results
            st.session_state.ml_data['X_train'] = X_train
            st.session_state.ml_data['X_test'] = X_test
            st.session_state.ml_data['y_train'] = y_train
            st.session_state.ml_data['y_test'] = y_test
            
            st.success("✅ Models trained successfully!")

    def _clustering_analysis(self, df):
        """Perform clustering analysis"""
        st.subheader("👥 Clustering Analysis")
        
        features = st.session_state.ml_data['features']
        X = df[features]
        
        # Clustering algorithms
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        
        clustering_method = st.selectbox(
            "Select Clustering Method:",
            ["K-Means", "DBSCAN", "Hierarchical"]
        )
        
        if clustering_method == "K-Means":
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
            
            if st.button("🚀 Perform Clustering"):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # Visualize clusters
                if len(features) >= 2:
                    fig = px.scatter(x=X[features[0]], y=X[features[1]], 
                                   color=clusters,
                                   title=f"K-Means Clustering (k={n_clusters})")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster summary
                cluster_summary = pd.DataFrame(X)
                cluster_summary['Cluster'] = clusters
                
                st.subheader("📊 Cluster Summary")
                summary_stats = cluster_summary.groupby('Cluster')[features].mean()
                st.dataframe(summary_stats, use_container_width=True)

    def model_evaluation(self):
        """Evaluate model performance"""
        st.subheader("📊 Model Performance Evaluation")
        
        if 'ml_data' not in st.session_state or 'training_results' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        results = st.session_state.ml_data['training_results']
        problem_type = st.session_state.ml_data['problem_type']
        
        # Model selection for evaluation
        model_names = list(results.keys())
        selected_model = st.selectbox("Select Model for Detailed Evaluation:", model_names)
        
        model_result = results[selected_model]
        y_test = model_result['y_test']
        y_pred = model_result['predictions']
        
        if problem_type == 'classification':
            self._classification_evaluation(selected_model, y_test, y_pred, model_result['model'])
        else:
            self._regression_evaluation(selected_model, y_test, y_pred)

    def _classification_evaluation(self, model_name, y_test, y_pred, model):
        """Evaluate classification model"""
        st.subheader(f"📊 {model_name} - Classification Evaluation")
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(st.session_state.ml_data['X_test'])[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                
                st.subheader("📈 ROC Curve")
                fig = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC = {auc:.3f})")
                fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                st.plotly_chart(fig, use_container_width=True)

    def _regression_evaluation(self, model_name, y_test, y_pred):
        """Evaluate regression model"""
        st.subheader(f"📊 {model_name} - Regression Evaluation")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{rmse:.3f}")
        with col2:
            st.metric("MAE", f"{mae:.3f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        with col4:
            st.metric("MSE", f"{mse:.3f}")
        
        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted Values")
        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), 
                     x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        residuals = y_test - y_pred
        st.subheader("📊 Residuals Analysis")
        fig = px.scatter(x=y_pred, y=residuals, title="Residuals Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    def hyperparameter_tuning(self):
        """Hyperparameter tuning and cross-validation"""
        st.subheader("📈 Cross-Validation & Hyperparameter Tuning")
        
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        models = st.session_state.ml_data['trained_models']
        X_train = st.session_state.ml_data['X_train']
        y_train = st.session_state.ml_data['y_train']
        
        # Model selection
        selected_model = st.selectbox("Select Model for Tuning:", list(models.keys()))
        
        # Cross-validation
        st.subheader("🔄 Cross-Validation")
        cv_folds = st.slider("Number of CV Folds:", 3, 10, 5)
        
        if st.button("🚀 Perform Cross-Validation"):
            model = models[selected_model]
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean CV Score", f"{cv_scores.mean():.3f}")
            with col2:
                st.metric("Std CV Score", f"{cv_scores.std():.3f}")
            with col3:
                st.metric("Best CV Score", f"{cv_scores.max():.3f}")
            
            # CV scores distribution
            fig = px.bar(x=range(1, len(cv_scores)+1), y=cv_scores,
                        title=f"Cross-Validation Scores - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Hyperparameter tuning
        st.subheader("⚙️ Hyperparameter Tuning")
        
        if selected_model == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif selected_model == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            st.info("🔧 Hyperparameter grids not defined for this model yet")
            return
        
        if st.button("🎯 Perform Grid Search"):
            model = models[selected_model]
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            st.write("**Best Parameters:**")
            st.json(grid_search.best_params_)
            
            st.metric("Best Cross-Validation Score", f"{grid_search.best_score_:.3f}")

    def feature_importance_analysis(self):
        """Analyze feature importance"""
        st.subheader("🔍 Feature Importance Analysis")
        
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        models = st.session_state.ml_data['trained_models']
        X_train = st.session_state.ml_data['X_train']
        
        # Model selection
        selected_model = st.selectbox("Select Model for Feature Analysis:", list(models.keys()))
        model = models[selected_model]
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Visualize top features
            top_n = st.slider("Number of Top Features to Show:", 5, len(feature_names), 10)
            top_features = importance_df.head(top_n)
            
            fig = px.bar(top_features, x='Importance', y='Feature',
                        orientation='h', title=f"Top {top_n} Feature Importances - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📊 Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
            
        else:
            st.info("🔧 Selected model doesn't provide feature importance scores")

    def model_comparison(self):
        """Compare multiple models"""
        st.subheader("📋 Model Comparison Dashboard")
        
        if 'ml_data' not in st.session_state or 'training_results' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        results = st.session_state.ml_data['training_results']
        problem_type = st.session_state.ml_data['problem_type']
        
        # Calculate metrics for all models
        comparison_data = []
        
        for model_name, result in results.items():
            y_test = result['y_test']
            y_pred = result['predictions']
            
            if problem_type == 'classification':
                metrics = {
                    'Model': model_name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted')
                }
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                metrics = {
                    'Model': model_name,
                    'RMSE': np.sqrt(mse),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R² Score': r2_score(y_test, y_pred),
                    'MSE': mse
                }
            
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model identification
        if problem_type == 'classification':
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
            best_score = comparison_df['Accuracy'].max()
            st.success(f"🏆 Best Model: {best_model} (Accuracy: {best_score:.3f})")
        else:
            best_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
            best_score = comparison_df['R² Score'].max()
            st.success(f"🏆 Best Model: {best_model} (R² Score: {best_score:.3f})")
        
        # Visualization
        if problem_type == 'classification':
            metric_col = st.selectbox("Select Metric to Visualize:", 
                                    ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        else:
            metric_col = st.selectbox("Select Metric to Visualize:", 
                                    ['R² Score', 'RMSE', 'MAE'])
        
        fig = px.bar(comparison_df, x='Model', y=metric_col,
                    title=f"Model Comparison - {metric_col}")
        st.plotly_chart(fig, use_container_width=True)

    def prediction_interface(self):
        """Interactive prediction interface"""
        st.subheader("🚀 Prediction Interface")
        
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        models = st.session_state.ml_data['trained_models']
        X_train = st.session_state.ml_data['X_train']
        
        # Model selection
        selected_model = st.selectbox("Select Model for Predictions:", list(models.keys()))
        model = models[selected_model]
        
        st.subheader("📝 Input Feature Values")
        
        # Create input fields for each feature
        input_data = {}
        
        for feature in X_train.columns:
            if X_train[feature].dtype in ['int64', 'float64']:
                min_val = float(X_train[feature].min())
                max_val = float(X_train[feature].max())
                mean_val = float(X_train[feature].mean())
                
                input_data[feature] = st.slider(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{feature}"
                )
            else:
                unique_values = X_train[feature].unique()
                input_data[feature] = st.selectbox(
                    f"{feature}:",
                    unique_values,
                    key=f"input_{feature}"
                )
        
        # Make prediction
        if st.button("🎯 Make Prediction"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            if st.session_state.ml_data['problem_type'] == 'classification':
                if 'label_encoder' in st.session_state.ml_data:
                    prediction = st.session_state.ml_data['label_encoder'].inverse_transform([prediction])[0]
                
                st.success(f"🎯 Predicted Class: {prediction}")
                
                # Show prediction probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_
                    
                    if 'label_encoder' in st.session_state.ml_data:
                        classes = st.session_state.ml_data['label_encoder'].inverse_transform(classes)
                    
                    prob_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    st.subheader("📊 Class Probabilities")
                    st.dataframe(prob_df, use_container_width=True)
            else:
                st.success(f"🎯 Predicted Value: {prediction:.2f}")

    def model_deployment_prep(self):
        """Prepare models for deployment"""
        st.subheader("💾 Model Export & Deployment Preparation")
        
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models first")
            return
        
        models = st.session_state.ml_data['trained_models']
        
        # Model selection for export
        selected_model = st.selectbox("Select Model to Export:", list(models.keys()))
        model = models[selected_model]
        
        st.subheader("📦 Model Information")
        
        # Model details
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Problem Type:** {st.session_state.ml_data['problem_type']}")
        st.write(f"**Number of Features:** {len(st.session_state.ml_data['X_train'].columns)}")
        
        # Feature list
        st.subheader("🎯 Required Features")
        features = st.session_state.ml_data['X_train'].columns.tolist()
        features_df = pd.DataFrame({
            'Feature Name': features,
            'Data Type': [str(st.session_state.ml_data['X_train'][col].dtype) for col in features]
        })
        st.dataframe(features_df, use_container_width=True)
        
        # Model serialization instructions
        st.subheader("💾 Export Instructions")
        
        st.code("""
import joblib
import pickle

# Save model using joblib (recommended)
joblib.dump(model, 'model.pkl')

# Or save using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model later
model = joblib.load('model.pkl')
        """, language='python')
        
        # API endpoint template
        st.subheader("🌐 FastAPI Deployment Template")
        
        api_code = f"""
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load('model.pkl')

class PredictionInput(BaseModel):
    {chr(10).join([f"    {feature}: float" for feature in features[:5]])}
    # ... add all features

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert to DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    return {{"prediction": prediction}}
        """
        
        st.code(api_code, language='python')

    def _load_datasets(self):
        """Load all available datasets"""
        try:
            import os
            
            # Product catalog
            product_file = os.path.join(self.data_path, "comprehensive_clothing_dataset_expanded_20250802_210102.csv")
            if os.path.exists(product_file):
                self.datasets['products'] = pd.read_csv(product_file)
            
            # Reviews dataset
            reviews_file = os.path.join(self.data_path, "zudio_reviews_dataset.csv")
            if os.path.exists(reviews_file):
                self.datasets['reviews'] = pd.read_csv(reviews_file)
            
            # Sales dataset
            sales_file = os.path.join(self.data_path, "Zudio_sales_data.csv")
            if os.path.exists(sales_file):
                self.datasets['sales'] = pd.read_csv(sales_file)
                
        except Exception as e:
            st.error(f"Error loading datasets: {str(e)}")
