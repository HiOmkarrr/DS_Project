"""
Experiment 3: EDA & Statistical Analysis
Comprehensive Exploratory Data Analysis with advanced visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Experiment3:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        self.data_path = r"C:\Users\Omkar\SEM_7\DS_Project\datasets\DS-2-8-25"
        self.datasets = {}
        
    def render_eda_dashboard(self):
        """Main EDA dashboard with comprehensive analysis"""
        st.markdown("""
        <div class="experiment-card">
            <h2>📊 Exploratory Data Analysis & Statistical Insights</h2>
            <p>Comprehensive exploration of fashion e-commerce data with advanced visualizations and statistical analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load datasets
        self._load_datasets()
        
        if not self.datasets:
            st.error("❌ No datasets found. Please ensure data files are available.")
            return
        
        # EDA Options
        eda_options = [
            "📊 Dataset Overview & Summary Statistics",
            "🎯 Univariate Analysis",
            "🔗 Bivariate & Correlation Analysis", 
            "📈 Customer Behavior Analysis",
            "💰 Sales & Pricing Analysis",
            "📝 Sentiment & Review Analysis",
            "🌟 Brand & Category Performance",
            "📅 Temporal Analysis",
            "🔍 Outlier Detection & Analysis",
            "📊 Advanced Statistical Insights"
        ]
        
        selected_analysis = st.selectbox(
            "Select EDA Analysis Type:",
            eda_options
        )
        
        # Route to appropriate analysis
        if selected_analysis == "📊 Dataset Overview & Summary Statistics":
            self.dataset_overview()
        elif selected_analysis == "🎯 Univariate Analysis":
            self.univariate_analysis()
        elif selected_analysis == "🔗 Bivariate & Correlation Analysis":
            self.bivariate_analysis()
        elif selected_analysis == "📈 Customer Behavior Analysis":
            self.customer_behavior_analysis()
        elif selected_analysis == "💰 Sales & Pricing Analysis":
            self.sales_pricing_analysis()
        elif selected_analysis == "📝 Sentiment & Review Analysis":
            self.sentiment_analysis()
        elif selected_analysis == "🌟 Brand & Category Performance":
            self.brand_category_analysis()
        elif selected_analysis == "📅 Temporal Analysis":
            self.temporal_analysis()
        elif selected_analysis == "🔍 Outlier Detection & Analysis":
            self.outlier_analysis()
        elif selected_analysis == "📊 Advanced Statistical Insights":
            self.advanced_statistical_analysis()

    def dataset_overview(self):
        """Comprehensive dataset overview and summary statistics"""
        st.subheader("📊 Dataset Overview & Summary Statistics")
        
        for name, df in self.datasets.items():
            with st.expander(f"📋 {name.title()} Dataset Overview", expanded=True):
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    st.metric("Missing Data %", f"{missing_pct:.1f}%")
                
                # Data types summary
                st.subheader("🏷️ Data Types Distribution")
                dtype_counts = df.dtypes.value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                               title=f"Data Types - {name.title()}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Column categories
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns
                    
                    st.write("**Column Categories:**")
                    st.write(f"• Numerical: {len(numerical_cols)} columns")
                    st.write(f"• Categorical: {len(categorical_cols)} columns") 
                    st.write(f"• DateTime: {len(datetime_cols)} columns")
                
                # Sample data
                st.subheader("🔍 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Descriptive statistics for numerical columns
                if len(numerical_cols) > 0:
                    st.subheader("📈 Numerical Statistics")
                    st.dataframe(df[numerical_cols].describe(), use_container_width=True)
                
                # Missing data analysis
                if df.isnull().sum().sum() > 0:
                    st.subheader("❓ Missing Data Analysis")
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    fig = px.bar(x=missing_data.index, y=missing_data.values,
                               title=f"Missing Values by Column - {name.title()}")
                    st.plotly_chart(fig, use_container_width=True)

    def univariate_analysis(self):
        """Detailed univariate analysis for each variable"""
        st.subheader("🎯 Univariate Analysis")
        
        # Dataset selection
        dataset_names = list(self.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset:", dataset_names)
        df = self.datasets[selected_dataset]
        
        # Variable selection
        all_columns = df.columns.tolist()
        selected_column = st.selectbox("Select Variable to Analyze:", all_columns)
        
        col_data = df[selected_column].dropna()
        
        if df[selected_column].dtype in ['int64', 'float64']:
            self._numerical_univariate_analysis(col_data, selected_column)
        else:
            self._categorical_univariate_analysis(col_data, selected_column)

    def _numerical_univariate_analysis(self, data, column_name):
        """Analyze numerical variables"""
        st.subheader(f"📊 Numerical Analysis: {column_name}")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{data.mean():.2f}")
        with col2:
            st.metric("Median", f"{data.median():.2f}")
        with col3:
            st.metric("Std Dev", f"{data.std():.2f}")
        with col4:
            st.metric("Range", f"{data.max() - data.min():.2f}")
        
        # Distribution visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot']
        )
        
        # Histogram
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Distribution'), row=1, col=1)
        
        # Box plot
        fig.add_trace(go.Box(y=data, name='Box Plot'), row=1, col=2)
        
        # Q-Q plot
        qq_data = stats.probplot(data, dist="norm")
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q'), row=2, col=1)
        
        # Violin plot
        fig.add_trace(go.Violin(y=data, name='Violin'), row=2, col=2)
        
        fig.update_layout(height=600, title=f"Distribution Analysis - {column_name}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical tests
        st.subheader("📊 Statistical Properties")
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        
        # Skewness and kurtosis
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        stats_df = pd.DataFrame({
            'Statistic': ['Skewness', 'Kurtosis', 'Shapiro-Wilk p-value', 'Is Normal (p>0.05)'],
            'Value': [f"{skewness:.3f}", f"{kurtosis:.3f}", f"{shapiro_p:.4f}", 
                     "Yes" if shapiro_p > 0.05 else "No"]
        })
        
        st.dataframe(stats_df, use_container_width=True)

    def _categorical_univariate_analysis(self, data, column_name):
        """Analyze categorical variables"""
        st.subheader(f"🏷️ Categorical Analysis: {column_name}")
        
        # Value counts
        value_counts = data.value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", len(value_counts))
        with col2:
            st.metric("Most Frequent", value_counts.index[0])
        with col3:
            st.metric("Mode Frequency", value_counts.iloc[0])
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(x=value_counts.index[:15], y=value_counts.values[:15],
                        title=f"Top 15 Categories - {column_name}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(values=value_counts.values[:10], names=value_counts.index[:10],
                        title=f"Top 10 Distribution - {column_name}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency table
        st.subheader("📊 Frequency Distribution")
        freq_df = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': (value_counts.values / len(data) * 100).round(2)
        })
        st.dataframe(freq_df.head(20), use_container_width=True)

    def bivariate_analysis(self):
        """Bivariate analysis and correlation exploration"""
        st.subheader("🔗 Bivariate & Correlation Analysis")
        
        # Dataset selection
        dataset_names = list(self.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset:", dataset_names, key="bivar_dataset")
        df = self.datasets[selected_dataset]
        
        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["📊 Numerical vs Numerical", "🏷️ Categorical vs Numerical", "🔗 Correlation Matrix"]
        )
        
        if analysis_type == "📊 Numerical vs Numerical":
            self._numerical_bivariate_analysis(df)
        elif analysis_type == "🏷️ Categorical vs Numerical":
            self._categorical_numerical_analysis(df)
        elif analysis_type == "🔗 Correlation Matrix":
            self._correlation_analysis(df)

    def _numerical_bivariate_analysis(self, df):
        """Analyze relationship between two numerical variables"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            st.warning("⚠️ Need at least 2 numerical columns for bivariate analysis")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select X Variable:", numerical_cols, key="x_var")
        with col2:
            y_var = st.selectbox("Select Y Variable:", [col for col in numerical_cols if col != x_var], key="y_var")
        
        # Clean data
        clean_data = df[[x_var, y_var]].dropna()
        
        # Scatter plot with regression line
        fig = px.scatter(clean_data, x=x_var, y=y_var, 
                        title=f"{x_var} vs {y_var}",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        correlation = clean_data[x_var].corr(clean_data[y_var])
        spearman_corr = clean_data[x_var].corr(clean_data[y_var], method='spearman')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pearson Correlation", f"{correlation:.3f}")
        with col2:
            st.metric("Spearman Correlation", f"{spearman_corr:.3f}")
        with col3:
            relationship = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
            st.metric("Relationship", relationship)

    def _categorical_numerical_analysis(self, df):
        """Analyze relationship between categorical and numerical variables"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numerical_cols) == 0 or len(categorical_cols) == 0:
            st.warning("⚠️ Need both numerical and categorical columns")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            cat_var = st.selectbox("Select Categorical Variable:", categorical_cols)
        with col2:
            num_var = st.selectbox("Select Numerical Variable:", numerical_cols)
        
        # Clean data
        clean_data = df[[cat_var, num_var]].dropna()
        
        # Box plot
        fig = px.box(clean_data, x=cat_var, y=num_var,
                    title=f"{num_var} by {cat_var}")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary by category
        summary_stats = clean_data.groupby(cat_var)[num_var].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        st.subheader("📊 Summary Statistics by Category")
        st.dataframe(summary_stats, use_container_width=True)

    def _correlation_analysis(self, df):
        """Comprehensive correlation analysis"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            st.warning("⚠️ Need at least 2 numerical columns for correlation analysis")
            return
        
        # Correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        st.subheader("🔗 Strong Correlations (|r| > 0.5)")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': f"{corr_val:.3f}",
                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            st.dataframe(strong_corr_df, use_container_width=True)
        else:
            st.info("No strong correlations found (|r| > 0.5)")

    def customer_behavior_analysis(self):
        """Analyze customer behavior patterns"""
        st.subheader("📈 Customer Behavior Analysis")
        
        # Check if we have reviews data
        if 'reviews' not in self.datasets:
            st.warning("⚠️ Reviews dataset not available for customer behavior analysis")
            return
        
        reviews_df = self.datasets['reviews']
        
        # Rating distribution
        if 'rating' in reviews_df.columns:
            st.subheader("⭐ Rating Distribution")
            
            rating_counts = reviews_df['rating'].value_counts().sort_index()
            
            fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                        title="Distribution of Customer Ratings")
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Rating", f"{reviews_df['rating'].mean():.2f}")
            with col2:
                st.metric("Median Rating", f"{reviews_df['rating'].median():.1f}")
            with col3:
                st.metric("Most Common", f"{reviews_df['rating'].mode().iloc[0]:.0f} stars")
            with col4:
                st.metric("Total Reviews", f"{len(reviews_df):,}")
        
        # Sentiment analysis if available
        if 'sentiment' in reviews_df.columns or 'sentiment_label' in reviews_df.columns:
            sentiment_col = 'sentiment' if 'sentiment' in reviews_df.columns else 'sentiment_label'
            
            st.subheader("😊 Sentiment Distribution")
            
            sentiment_counts = reviews_df[sentiment_col].value_counts()
            
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Customer Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Review length analysis
        if 'review_text' in reviews_df.columns:
            st.subheader("📝 Review Length Analysis")
            
            reviews_df['review_length'] = reviews_df['review_text'].astype(str).str.len()
            
            fig = px.histogram(reviews_df, x='review_length', nbins=50,
                              title="Distribution of Review Lengths")
            st.plotly_chart(fig, use_container_width=True)

    def sales_pricing_analysis(self):
        """Analyze sales patterns and pricing strategies"""
        st.subheader("💰 Sales & Pricing Analysis")
        
        # Check available datasets
        available_datasets = []
        if 'products' in self.datasets:
            available_datasets.append('products')
        if 'sales' in self.datasets:
            available_datasets.append('sales')
        
        if not available_datasets:
            st.warning("⚠️ No product or sales data available")
            return
        
        # Price analysis from products dataset
        if 'products' in self.datasets:
            products_df = self.datasets['products']
            
            if 'price' in products_df.columns:
                st.subheader("💵 Price Distribution Analysis")
                
                # Price statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Price", f"${products_df['price'].mean():.2f}")
                with col2:
                    st.metric("Median Price", f"${products_df['price'].median():.2f}")
                with col3:
                    st.metric("Price Range", f"${products_df['price'].min():.0f} - ${products_df['price'].max():.0f}")
                with col4:
                    st.metric("Total Products", f"{len(products_df):,}")
                
                # Price distribution
                fig = px.histogram(products_df, x='price', nbins=50,
                                  title="Product Price Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Price by category
                if 'category' in products_df.columns:
                    st.subheader("📊 Price by Category")
                    
                    fig = px.box(products_df, x='category', y='price',
                                title="Price Distribution by Category")
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Discount analysis
                if 'original_price' in products_df.columns:
                    products_df['discount_amount'] = products_df['original_price'] - products_df['price']
                    products_df['discount_percentage'] = (products_df['discount_amount'] / products_df['original_price']) * 100
                    
                    st.subheader("🏷️ Discount Analysis")
                    
                    fig = px.histogram(products_df, x='discount_percentage', nbins=30,
                                      title="Distribution of Discount Percentages")
                    st.plotly_chart(fig, use_container_width=True)

    def sentiment_analysis(self):
        """Analyze sentiment patterns in reviews"""
        st.subheader("📝 Sentiment & Review Analysis")
        
        if 'reviews' not in self.datasets:
            st.warning("⚠️ Reviews dataset not available")
            return
        
        reviews_df = self.datasets['reviews']
        
        # Sentiment distribution
        sentiment_cols = [col for col in reviews_df.columns if 'sentiment' in col.lower()]
        
        if sentiment_cols:
            sentiment_col = sentiment_cols[0]
            
            st.subheader("😊 Sentiment Analysis Overview")
            
            sentiment_counts = reviews_df[sentiment_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment metrics
                total_reviews = len(reviews_df)
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_reviews) * 100
                    st.metric(f"{sentiment.title()} Reviews", f"{count:,} ({percentage:.1f}%)")
        
        # Rating vs Sentiment correlation
        if 'rating' in reviews_df.columns and sentiment_cols:
            st.subheader("⭐ Rating vs Sentiment Correlation")
            
            # Create cross-tabulation
            crosstab = pd.crosstab(reviews_df['rating'], reviews_df[sentiment_col])
            
            fig = px.imshow(crosstab.values, 
                           x=crosstab.columns, 
                           y=crosstab.index,
                           title="Rating vs Sentiment Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    def brand_category_analysis(self):
        """Analyze brand and category performance"""
        st.subheader("🌟 Brand & Category Performance")
        
        if 'products' not in self.datasets:
            st.warning("⚠️ Products dataset not available")
            return
        
        products_df = self.datasets['products']
        
        # Category analysis
        if 'category' in products_df.columns:
            st.subheader("📊 Category Performance")
            
            category_counts = products_df['category'].value_counts()
            
            fig = px.bar(x=category_counts.index, y=category_counts.values,
                        title="Number of Products by Category")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Brand analysis
        if 'brand' in products_df.columns:
            st.subheader("🏷️ Brand Performance")
            
            brand_counts = products_df['brand'].value_counts().head(15)
            
            fig = px.bar(x=brand_counts.values, y=brand_counts.index,
                        orientation='h',
                        title="Top 15 Brands by Product Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Brand-Category matrix
            if 'category' in products_df.columns:
                st.subheader("🎯 Brand-Category Matrix")
                
                brand_category = pd.crosstab(products_df['brand'], products_df['category'])
                
                # Show top brands only
                top_brands = products_df['brand'].value_counts().head(10).index
                brand_category_top = brand_category.loc[top_brands]
                
                fig = px.imshow(brand_category_top.values,
                               x=brand_category_top.columns,
                               y=brand_category_top.index,
                               title="Brand-Category Distribution (Top 10 Brands)")
                st.plotly_chart(fig, use_container_width=True)

    def temporal_analysis(self):
        """Analyze temporal patterns in the data"""
        st.subheader("📅 Temporal Analysis")
        
        # Check for date columns
        date_columns = []
        for name, df in self.datasets.items():
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_columns.append((name, col))
        
        if not date_columns:
            st.warning("⚠️ No date/time columns found in datasets")
            return
        
        # Select date column
        date_options = [f"{dataset} - {col}" for dataset, col in date_columns]
        selected_date = st.selectbox("Select Date Column:", date_options)
        
        dataset_name, col_name = selected_date.split(" - ")
        df = self.datasets[dataset_name]
        
        # Convert to datetime
        try:
            df[col_name] = pd.to_datetime(df[col_name])
            
            # Temporal patterns
            st.subheader("📈 Temporal Patterns")
            
            # Daily pattern
            df['day_of_week'] = df[col_name].dt.day_name()
            day_counts = df['day_of_week'].value_counts()
            
            fig = px.bar(x=day_counts.index, y=day_counts.values,
                        title="Activity by Day of Week")
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly pattern
            df['month'] = df[col_name].dt.month_name()
            month_counts = df['month'].value_counts()
            
            fig = px.bar(x=month_counts.index, y=month_counts.values,
                        title="Activity by Month")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing date column: {str(e)}")

    def outlier_analysis(self):
        """Detect and analyze outliers in numerical data"""
        st.subheader("🔍 Outlier Detection & Analysis")
        
        # Dataset selection
        dataset_names = list(self.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset:", dataset_names, key="outlier_dataset")
        df = self.datasets[selected_dataset]
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            st.warning("⚠️ No numerical columns found for outlier analysis")
            return
        
        selected_column = st.selectbox("Select Column for Outlier Analysis:", numerical_cols)
        
        col_data = df[selected_column].dropna()
        
        # Outlier detection methods
        st.subheader("📊 Outlier Detection Methods")
        
        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data[z_scores > 3]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Values", len(col_data))
        with col2:
            st.metric("IQR Outliers", len(iqr_outliers))
        with col3:
            st.metric("Z-Score Outliers", len(z_outliers))
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Box Plot with Outliers', 'Histogram with Outlier Bounds']
        )
        
        # Box plot
        fig.add_trace(go.Box(y=col_data, name='Data'), row=1, col=1)
        
        # Histogram
        fig.add_trace(go.Histogram(x=col_data, nbinsx=50, name='Distribution'), row=1, col=2)
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_layout(height=400, title=f"Outlier Analysis - {selected_column}")
        st.plotly_chart(fig, use_container_width=True)

    def advanced_statistical_analysis(self):
        """Advanced statistical insights and tests"""
        st.subheader("📊 Advanced Statistical Insights")
        
        if not self.datasets:
            st.warning("No datasets loaded. Please load datasets first.")
            return
        
        # Dataset selection
        dataset_name = st.selectbox("Select Dataset for Analysis:", list(self.datasets.keys()))
        df = self.datasets[dataset_name]
        
        # Import statistical analysis module
        try:
            from experiments.statistical_analysis import StatisticalAnalysis
            
            stats_analyzer = StatisticalAnalysis(self.datasets)
            stats_analyzer.render_statistical_dashboard()
        except ImportError:
            # Fallback to basic statistical analysis
            st.info("Using basic statistical analysis...")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numerical_cols:
                st.warning("No numerical columns found for statistical analysis.")
                return
            
            # Normality tests
            st.subheader("🔍 Normality Tests")
            test_column = st.selectbox("Select Column for Normality Test:", numerical_cols)
            
            col_data = df[test_column].dropna()
            
            # Shapiro-Wilk test
            if len(col_data) <= 5000:  # Shapiro-Wilk has sample size limitation
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                st.write(f"**Shapiro-Wilk Test:**")
                st.write(f"- Statistic: {shapiro_stat:.4f}")
                st.write(f"- p-value: {shapiro_p:.4f}")
                st.write(f"- Normal Distribution: {'Yes' if shapiro_p > 0.05 else 'No'}")
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
            st.write(f"**Kolmogorov-Smirnov Test:**")
            st.write(f"- Statistic: {ks_stat:.4f}")
            st.write(f"- p-value: {ks_p:.4f}")
            st.write(f"- Normal Distribution: {'Yes' if ks_p > 0.05 else 'No'}")
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            summary_stats = df[numerical_cols].describe()
            st.dataframe(summary_stats)
            
            # Hypothesis testing
            st.subheader("🧪 Hypothesis Testing")
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Variable 1:", numerical_cols, key="var1")
                with col2:
                    var2 = st.selectbox("Variable 2:", numerical_cols, key="var2", index=1)
                
                if var1 != var2:
                    data1 = df[var1].dropna()
                    data2 = df[var2].dropna()
                    
                    # T-test
                    t_stat, t_p = stats.ttest_ind(data1, data2)
                    st.write(f"**Independent T-Test ({var1} vs {var2}):**")
                    st.write(f"- T-statistic: {t_stat:.4f}")
                    st.write(f"- p-value: {t_p:.4f}")
                    st.write(f"- Significant Difference: {'Yes' if t_p < 0.05 else 'No'}")
                    
                    # Correlation
                    correlation = df[[var1, var2]].corr().iloc[0, 1]
                    st.write(f"**Correlation:** {correlation:.4f}")
                    
                    # Visualization
                    fig = px.scatter(df, x=var1, y=var2, title=f"{var1} vs {var2}")
                    st.plotly_chart(fig, use_container_width=True)

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
