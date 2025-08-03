"""
Statistical Analysis Module for Data Science Project
Comprehensive mathematical properties and statistical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, normaltest, jarque_bera
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
        
    def render_statistical_dashboard(self):
        """Main statistical analysis dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>📊 Mathematical Properties & Statistical Analysis</h2>
            <p>Comprehensive analysis of statistical properties, relationships, and mathematical characteristics of the data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistical analysis categories
        analysis_categories = [
            "🔢 Descriptive Statistics",
            "📈 Distribution Analysis", 
            "🔗 Independence & Correlation Analysis",
            "⚖️ Mutual Exclusivity Analysis",
            "📐 Skewness & Kurtosis Analysis",
            "🎯 Normality Testing",
            "🔄 Stationarity Analysis",
            "📊 Heteroscedasticity Testing",
            "🧮 Central Limit Theorem Validation",
            "🎲 Randomness & Entropy Analysis",
            "📏 Distance & Similarity Metrics",
            "🌐 Multicollinearity Analysis",
            "🔀 Principal Component Analysis"
        ]
        
        selected_analysis = st.selectbox(
            "Select Statistical Analysis Category:",
            analysis_categories
        )
        
        # Route to appropriate analysis
        if selected_analysis == "🔢 Descriptive Statistics":
            self.descriptive_statistics()
        elif selected_analysis == "📈 Distribution Analysis":
            self.distribution_analysis()
        elif selected_analysis == "🔗 Independence & Correlation Analysis":
            self.independence_correlation_analysis()
        elif selected_analysis == "⚖️ Mutual Exclusivity Analysis":
            self.mutual_exclusivity_analysis()
        elif selected_analysis == "📐 Skewness & Kurtosis Analysis":
            self.skewness_kurtosis_analysis()
        elif selected_analysis == "🎯 Normality Testing":
            self.normality_testing()
        elif selected_analysis == "🔄 Stationarity Analysis":
            self.stationarity_analysis()
        elif selected_analysis == "📊 Heteroscedasticity Testing":
            self.heteroscedasticity_testing()
        elif selected_analysis == "🧮 Central Limit Theorem Validation":
            self.central_limit_theorem()
        elif selected_analysis == "🎲 Randomness & Entropy Analysis":
            self.randomness_entropy_analysis()
        elif selected_analysis == "📏 Distance & Similarity Metrics":
            self.distance_similarity_metrics()
        elif selected_analysis == "🌐 Multicollinearity Analysis":
            self.multicollinearity_analysis()
        elif selected_analysis == "🔀 Principal Component Analysis":
            self.principal_component_analysis()

    def descriptive_statistics(self):
        """Comprehensive descriptive statistics analysis"""
        st.subheader("🔢 Descriptive Statistics Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"📊 {name.title()} Dataset Statistics", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    # Basic descriptive statistics
                    st.write("**📈 Basic Descriptive Statistics**")
                    desc_stats = df[numerical_cols].describe()
                    
                    # Add additional statistics
                    additional_stats = pd.DataFrame({
                        col: {
                            'variance': df[col].var(),
                            'skewness': df[col].skew(),
                            'kurtosis': df[col].kurtosis(),
                            'range': df[col].max() - df[col].min(),
                            'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                            'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan,
                            'mad': df[col].mad(),  # Mean Absolute Deviation
                            'sem': df[col].sem()   # Standard Error of Mean
                        }
                        for col in numerical_cols
                    }).T
                    
                    combined_stats = pd.concat([desc_stats, additional_stats])
                    st.dataframe(combined_stats, use_container_width=True)
                    
                    # Moment-based statistics visualization
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Mean vs Median', 'Variance vs Standard Deviation', 
                                      'Skewness Distribution', 'Kurtosis Distribution']
                    )
                    
                    # Mean vs Median
                    means = [df[col].mean() for col in numerical_cols]
                    medians = [df[col].median() for col in numerical_cols]
                    fig.add_trace(go.Scatter(x=means, y=medians, mode='markers+text',
                                           text=numerical_cols, name='Mean vs Median'),
                                row=1, col=1)
                    
                    # Variance vs Std Dev
                    variances = [df[col].var() for col in numerical_cols]
                    std_devs = [df[col].std() for col in numerical_cols]
                    fig.add_trace(go.Scatter(x=variances, y=std_devs, mode='markers+text',
                                           text=numerical_cols, name='Var vs Std'),
                                row=1, col=2)
                    
                    # Skewness
                    skewness_values = [df[col].skew() for col in numerical_cols]
                    fig.add_trace(go.Bar(x=numerical_cols, y=skewness_values, name='Skewness'),
                                row=2, col=1)
                    
                    # Kurtosis
                    kurtosis_values = [df[col].kurtosis() for col in numerical_cols]
                    fig.add_trace(go.Bar(x=numerical_cols, y=kurtosis_values, name='Kurtosis'),
                                row=2, col=2)
                    
                    fig.update_layout(height=600, title=f"Statistical Moments - {name.title()}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical significance tests
                    st.write("**🎯 Statistical Insights**")
                    for col in numerical_cols:
                        col_data = df[col].dropna()
                        mean_val = col_data.mean()
                        median_val = col_data.median()
                        skew_val = col_data.skew()
                        
                        insights = []
                        if abs(mean_val - median_val) > 0.1 * col_data.std():
                            insights.append("Distribution is asymmetric")
                        if abs(skew_val) > 1:
                            insights.append("Highly skewed distribution")
                        elif abs(skew_val) > 0.5:
                            insights.append("Moderately skewed distribution")
                        else:
                            insights.append("Approximately symmetric distribution")
                        
                        st.write(f"**{col}**: {', '.join(insights)}")

    def distribution_analysis(self):
        """Analyze data distributions and their properties"""
        st.subheader("📈 Distribution Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"📊 {name.title()} Distribution Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    selected_col = st.selectbox(f"Select column for distribution analysis ({name}):", 
                                               numerical_cols, key=f"dist_col_{name}")
                    
                    col_data = df[selected_col].dropna()
                    
                    # Distribution visualization
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Histogram with KDE', 'Q-Q Plot', 'Box Plot', 'Violin Plot']
                    )
                    
                    # Histogram with KDE
                    fig.add_trace(go.Histogram(x=col_data, nbinsx=30, name='Histogram', 
                                             histnorm='probability density'), row=1, col=1)
                    
                    # Q-Q Plot
                    qq_data = stats.probplot(col_data, dist="norm")
                    fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                           mode='markers', name='Q-Q Plot'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0]*qq_data[0][0],
                                           mode='lines', name='Reference Line'), row=1, col=2)
                    
                    # Box Plot
                    fig.add_trace(go.Box(y=col_data, name='Box Plot'), row=2, col=1)
                    
                    # Violin Plot
                    fig.add_trace(go.Violin(y=col_data, name='Violin Plot'), row=2, col=2)
                    
                    fig.update_layout(height=600, title=f"Distribution Analysis - {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution parameters
                    st.write("**📊 Distribution Parameters**")
                    
                    # Test for different distributions
                    distributions = ['norm', 'expon', 'gamma', 'lognorm', 'beta']
                    best_fit_results = []
                    
                    for dist_name in distributions:
                        try:
                            dist = getattr(stats, dist_name)
                            params = dist.fit(col_data)
                            ks_stat, ks_p = stats.kstest(col_data, lambda x: dist.cdf(x, *params))
                            best_fit_results.append({
                                'Distribution': dist_name,
                                'KS Statistic': ks_stat,
                                'P-value': ks_p,
                                'Parameters': str(params)
                            })
                        except:
                            continue
                    
                    if best_fit_results:
                        fit_df = pd.DataFrame(best_fit_results).sort_values('KS Statistic')
                        st.dataframe(fit_df, use_container_width=True)
                        
                        best_dist = fit_df.iloc[0]['Distribution']
                        st.success(f"🎯 Best fitting distribution: **{best_dist}**")

    def independence_correlation_analysis(self):
        """Analyze independence and correlations between variables"""
        st.subheader("🔗 Independence & Correlation Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"🔗 {name.title()} Independence Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(numerical_cols) > 1:
                    # Correlation matrix
                    st.write("**📊 Pearson Correlation Matrix**")
                    corr_matrix = df[numerical_cols].corr()
                    
                    fig = px.imshow(corr_matrix, 
                                   title="Pearson Correlation Heatmap",
                                   color_continuous_scale="RdBu_r",
                                   aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Spearman correlation for monotonic relationships
                    st.write("**📈 Spearman Rank Correlation Matrix**")
                    spearman_corr = df[numerical_cols].corr(method='spearman')
                    
                    fig = px.imshow(spearman_corr,
                                   title="Spearman Correlation Heatmap", 
                                   color_continuous_scale="RdBu_r",
                                   aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical significance of correlations
                    st.write("**🎯 Correlation Significance Tests**")
                    correlation_tests = []
                    
                    for i, col1 in enumerate(numerical_cols):
                        for j, col2 in enumerate(numerical_cols):
                            if i < j:  # Avoid duplicates
                                data1 = df[col1].dropna()
                                data2 = df[col2].dropna()
                                
                                # Align the data
                                common_idx = data1.index.intersection(data2.index)
                                data1_aligned = data1.loc[common_idx]
                                data2_aligned = data2.loc[common_idx]
                                
                                if len(data1_aligned) > 10:  # Minimum sample size
                                    # Pearson correlation
                                    pearson_r, pearson_p = pearsonr(data1_aligned, data2_aligned)
                                    
                                    # Spearman correlation
                                    spearman_r, spearman_p = spearmanr(data1_aligned, data2_aligned)
                                    
                                    correlation_tests.append({
                                        'Variable 1': col1,
                                        'Variable 2': col2,
                                        'Pearson r': f"{pearson_r:.3f}",
                                        'Pearson p-value': f"{pearson_p:.4f}",
                                        'Spearman ρ': f"{spearman_r:.3f}",
                                        'Spearman p-value': f"{spearman_p:.4f}",
                                        'Independence': 'Dependent' if min(pearson_p, spearman_p) < 0.05 else 'Independent'
                                    })
                    
                    if correlation_tests:
                        corr_df = pd.DataFrame(correlation_tests)
                        st.dataframe(corr_df, use_container_width=True)
                
                # Chi-square test for categorical independence
                if len(categorical_cols) > 1:
                    st.write("**🔄 Categorical Independence Tests (Chi-square)**")
                    chi_square_tests = []
                    
                    for i, col1 in enumerate(categorical_cols):
                        for j, col2 in enumerate(categorical_cols):
                            if i < j:
                                # Create contingency table
                                contingency_table = pd.crosstab(df[col1], df[col2])
                                
                                if contingency_table.size > 1:
                                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                                    
                                    chi_square_tests.append({
                                        'Variable 1': col1,
                                        'Variable 2': col2,
                                        'Chi-square': f"{chi2:.3f}",
                                        'P-value': f"{p_value:.4f}",
                                        'Degrees of Freedom': dof,
                                        'Independence': 'Independent' if p_value >= 0.05 else 'Dependent'
                                    })
                    
                    if chi_square_tests:
                        chi_df = pd.DataFrame(chi_square_tests)
                        st.dataframe(chi_df, use_container_width=True)

    def mutual_exclusivity_analysis(self):
        """Analyze mutual exclusivity between events/categories"""
        st.subheader("⚖️ Mutual Exclusivity Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"⚖️ {name.title()} Mutual Exclusivity", expanded=True):
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(categorical_cols) > 0:
                    st.write("**🔍 Mutual Exclusivity Between Categories**")
                    
                    for col in categorical_cols:
                        if df[col].nunique() > 1:
                            value_counts = df[col].value_counts()
                            
                            st.write(f"**Analysis for {col}:**")
                            
                            # Check if categories are mutually exclusive
                            total_records = len(df)
                            sum_categories = value_counts.sum()
                            
                            # Mutual exclusivity metrics
                            overlap_analysis = {
                                'Total Records': total_records,
                                'Sum of Categories': sum_categories,
                                'Mutual Exclusivity': 'Perfect' if total_records == sum_categories else 'Partial',
                                'Overlap Percentage': f"{((sum_categories - total_records) / total_records * 100):.2f}%" if sum_categories > total_records else "0%"
                            }
                            
                            st.json(overlap_analysis)
                            
                            # Visualization
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                        title=f"Category Distribution - {col}")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Set-based mutual exclusivity analysis
                if len(categorical_cols) > 1:
                    st.write("**🔄 Set-based Mutual Exclusivity Analysis**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("Select first variable:", categorical_cols, key=f"mut_ex_1_{name}")
                    with col2:
                        var2 = st.selectbox("Select second variable:", categorical_cols, key=f"mut_ex_2_{name}")
                    
                    if var1 != var2:
                        # Create Venn diagram analysis
                        set1_values = set(df[var1].dropna().unique())
                        set2_values = set(df[var2].dropna().unique())
                        
                        intersection = set1_values.intersection(set2_values)
                        union = set1_values.union(set2_values)
                        
                        venn_analysis = {
                            'Set 1 Size': len(set1_values),
                            'Set 2 Size': len(set2_values),
                            'Intersection Size': len(intersection),
                            'Union Size': len(union),
                            'Jaccard Index': len(intersection) / len(union) if len(union) > 0 else 0,
                            'Mutual Exclusivity': 'Mutually Exclusive' if len(intersection) == 0 else 'Not Mutually Exclusive'
                        }
                        
                        st.json(venn_analysis)

    def skewness_kurtosis_analysis(self):
        """Analyze skewness and kurtosis of distributions"""
        st.subheader("📐 Skewness & Kurtosis Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"📐 {name.title()} Skewness & Kurtosis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    # Calculate skewness and kurtosis
                    skew_kurt_data = []
                    
                    for col in numerical_cols:
                        col_data = df[col].dropna()
                        if len(col_data) > 3:  # Minimum sample size
                            skewness = col_data.skew()
                            kurtosis = col_data.kurtosis()
                            
                            # Interpretation
                            if abs(skewness) < 0.5:
                                skew_interp = "Approximately Symmetric"
                            elif abs(skewness) < 1:
                                skew_interp = "Moderately Skewed"
                            else:
                                skew_interp = "Highly Skewed"
                            
                            if skewness > 0:
                                skew_direction = "Right-skewed (Positive)"
                            elif skewness < 0:
                                skew_direction = "Left-skewed (Negative)"
                            else:
                                skew_direction = "Symmetric"
                            
                            if kurtosis < 0:
                                kurt_interp = "Platykurtic (Light-tailed)"
                            elif kurtosis > 0:
                                kurt_interp = "Leptokurtic (Heavy-tailed)"
                            else:
                                kurt_interp = "Mesokurtic (Normal-like)"
                            
                            skew_kurt_data.append({
                                'Variable': col,
                                'Skewness': f"{skewness:.3f}",
                                'Skewness Interpretation': skew_interp,
                                'Skewness Direction': skew_direction,
                                'Kurtosis': f"{kurtosis:.3f}",
                                'Kurtosis Interpretation': kurt_interp
                            })
                    
                    if skew_kurt_data:
                        skew_kurt_df = pd.DataFrame(skew_kurt_data)
                        st.dataframe(skew_kurt_df, use_container_width=True)
                        
                        # Visualization
                        skewness_values = [float(row['Skewness']) for row in skew_kurt_data]
                        kurtosis_values = [float(row['Kurtosis']) for row in skew_kurt_data]
                        variable_names = [row['Variable'] for row in skew_kurt_data]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=skewness_values,
                            y=kurtosis_values,
                            mode='markers+text',
                            text=variable_names,
                            textposition="top center",
                            marker=dict(size=10, color='blue'),
                            name='Variables'
                        ))
                        
                        # Add reference lines
                        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                                     annotation_text="Normal Kurtosis")
                        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                                     annotation_text="No Skewness")
                        
                        fig.update_layout(
                            title="Skewness vs Kurtosis Plot",
                            xaxis_title="Skewness",
                            yaxis_title="Kurtosis",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

    def normality_testing(self):
        """Test for normality using multiple statistical tests"""
        st.subheader("🎯 Normality Testing")
        
        for name, df in self.datasets.items():
            with st.expander(f"🎯 {name.title()} Normality Tests", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    normality_results = []
                    
                    for col in numerical_cols:
                        col_data = df[col].dropna()
                        
                        if len(col_data) > 8:  # Minimum sample size for tests
                            # Shapiro-Wilk Test (best for small samples)
                            if len(col_data) <= 5000:
                                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                            else:
                                shapiro_stat, shapiro_p = np.nan, np.nan
                            
                            # D'Agostino-Pearson Test
                            dagostino_stat, dagostino_p = normaltest(col_data)
                            
                            # Jarque-Bera Test
                            jb_stat, jb_p = jarque_bera(col_data)
                            
                            # Kolmogorov-Smirnov Test
                            ks_stat, ks_p = stats.kstest(col_data, 'norm', 
                                                        args=(col_data.mean(), col_data.std()))
                            
                            # Anderson-Darling Test
                            ad_result = stats.anderson(col_data, dist='norm')
                            ad_stat = ad_result.statistic
                            ad_critical = ad_result.critical_values[2]  # 5% significance level
                            ad_normal = "Yes" if ad_stat < ad_critical else "No"
                            
                            normality_results.append({
                                'Variable': col,
                                'Shapiro-Wilk p-value': f"{shapiro_p:.4f}" if not np.isnan(shapiro_p) else "N/A",
                                'Shapiro Normal': "Yes" if not np.isnan(shapiro_p) and shapiro_p > 0.05 else "No",
                                'D\'Agostino p-value': f"{dagostino_p:.4f}",
                                'D\'Agostino Normal': "Yes" if dagostino_p > 0.05 else "No",
                                'Jarque-Bera p-value': f"{jb_p:.4f}",
                                'JB Normal': "Yes" if jb_p > 0.05 else "No",
                                'KS p-value': f"{ks_p:.4f}",
                                'KS Normal': "Yes" if ks_p > 0.05 else "No",
                                'Anderson-Darling Normal': ad_normal,
                                'Overall Assessment': self._assess_normality(shapiro_p, dagostino_p, jb_p, ks_p, ad_normal)
                            })
                    
                    if normality_results:
                        norm_df = pd.DataFrame(normality_results)
                        st.dataframe(norm_df, use_container_width=True)
                        
                        # Summary of normality
                        normal_count = sum(1 for result in normality_results 
                                         if result['Overall Assessment'] == 'Normal')
                        total_variables = len(normality_results)
                        
                        st.metric("Variables Following Normal Distribution", 
                                f"{normal_count}/{total_variables} ({normal_count/total_variables*100:.1f}%)")

    def _assess_normality(self, shapiro_p, dagostino_p, jb_p, ks_p, ad_normal):
        """Assess overall normality based on multiple tests"""
        normal_tests = 0
        total_tests = 0
        
        if not np.isnan(shapiro_p):
            normal_tests += 1 if shapiro_p > 0.05 else 0
            total_tests += 1
        
        normal_tests += 1 if dagostino_p > 0.05 else 0
        normal_tests += 1 if jb_p > 0.05 else 0
        normal_tests += 1 if ks_p > 0.05 else 0
        normal_tests += 1 if ad_normal == "Yes" else 0
        total_tests += 4
        
        if normal_tests >= total_tests * 0.6:  # 60% of tests suggest normality
            return "Normal"
        else:
            return "Non-Normal"

    def stationarity_analysis(self):
        """Analyze stationarity for time series data"""
        st.subheader("🔄 Stationarity Analysis")
        st.info("This analysis is most relevant for time series data. For non-time series data, results may not be meaningful.")
        
        for name, df in self.datasets.items():
            with st.expander(f"🔄 {name.title()} Stationarity Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    selected_col = st.selectbox(f"Select variable for stationarity test ({name}):",
                                               numerical_cols, key=f"stat_col_{name}")
                    
                    col_data = df[selected_col].dropna()
                    
                    if len(col_data) > 10:
                        # Simple stationarity checks
                        # Rolling statistics
                        window_size = min(len(col_data) // 4, 20)
                        rolling_mean = col_data.rolling(window=window_size).mean()
                        rolling_std = col_data.rolling(window=window_size).std()
                        
                        # Visualization
                        fig = make_subplots(rows=3, cols=1,
                                          subplot_titles=['Original Series', 'Rolling Mean', 'Rolling Std'])
                        
                        fig.add_trace(go.Scatter(y=col_data.values, name='Original'), row=1, col=1)
                        fig.add_trace(go.Scatter(y=rolling_mean.values, name='Rolling Mean'), row=2, col=1)
                        fig.add_trace(go.Scatter(y=rolling_std.values, name='Rolling Std'), row=3, col=1)
                        
                        fig.update_layout(height=600, title=f"Stationarity Analysis - {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Simple stationarity metrics
                        mean_change = abs(rolling_mean.iloc[-1] - rolling_mean.iloc[window_size]) / rolling_mean.iloc[window_size] * 100
                        std_change = abs(rolling_std.iloc[-1] - rolling_std.iloc[window_size]) / rolling_std.iloc[window_size] * 100
                        
                        stationarity_assessment = {
                            'Mean Change (%)': f"{mean_change:.2f}%",
                            'Std Change (%)': f"{std_change:.2f}%",
                            'Likely Stationary': "Yes" if mean_change < 10 and std_change < 20 else "No"
                        }
                        
                        st.json(stationarity_assessment)

    def heteroscedasticity_testing(self):
        """Test for heteroscedasticity in the data"""
        st.subheader("📊 Heteroscedasticity Testing")
        
        for name, df in self.datasets.items():
            with st.expander(f"📊 {name.title()} Heteroscedasticity Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 1:
                    col1, col2 = st.columns(2)
                    with col1:
                        dependent_var = st.selectbox(f"Select dependent variable ({name}):", 
                                                   numerical_cols, key=f"het_dep_{name}")
                    with col2:
                        independent_var = st.selectbox(f"Select independent variable ({name}):", 
                                                     [col for col in numerical_cols if col != dependent_var], 
                                                     key=f"het_indep_{name}")
                    
                    # Prepare data
                    clean_data = df[[dependent_var, independent_var]].dropna()
                    
                    if len(clean_data) > 10:
                        x = clean_data[independent_var]
                        y = clean_data[dependent_var]
                        
                        # Linear regression residuals
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        X_reshaped = x.values.reshape(-1, 1)
                        model.fit(X_reshaped, y)
                        y_pred = model.predict(X_reshaped)
                        residuals = y - y_pred
                        
                        # Visualization
                        fig = make_subplots(rows=1, cols=2,
                                          subplot_titles=['Residuals vs Fitted', 'Residuals vs Independent Var'])
                        
                        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                               name='Residuals vs Fitted'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=x, y=residuals, mode='markers',
                                               name='Residuals vs X'), row=1, col=2)
                        
                        fig.update_layout(height=400, title="Heteroscedasticity Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Simple heteroscedasticity test
                        # Breusch-Pagan-like test (simplified)
                        residuals_abs = np.abs(residuals)
                        correlation, p_value = pearsonr(x, residuals_abs)
                        
                        het_results = {
                            'Correlation (|residuals| vs X)': f"{correlation:.4f}",
                            'P-value': f"{p_value:.4f}",
                            'Heteroscedasticity Present': "Yes" if p_value < 0.05 else "No",
                            'Variance': "Constant" if p_value >= 0.05 else "Non-constant"
                        }
                        
                        st.json(het_results)

    def central_limit_theorem(self):
        """Demonstrate and validate Central Limit Theorem"""
        st.subheader("🧮 Central Limit Theorem Validation")
        
        for name, df in self.datasets.items():
            with st.expander(f"🧮 {name.title()} CLT Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 0:
                    selected_col = st.selectbox(f"Select variable for CLT analysis ({name}):",
                                               numerical_cols, key=f"clt_col_{name}")
                    
                    col_data = df[selected_col].dropna()
                    
                    if len(col_data) > 30:
                        # CLT simulation
                        sample_sizes = [5, 10, 30, 50]
                        n_samples = 1000
                        
                        fig = make_subplots(rows=2, cols=2,
                                          subplot_titles=[f'Sample Size = {n}' for n in sample_sizes])
                        
                        clt_results = []
                        
                        for i, sample_size in enumerate(sample_sizes):
                            sample_means = []
                            for _ in range(n_samples):
                                sample = np.random.choice(col_data, size=sample_size, replace=True)
                                sample_means.append(np.mean(sample))
                            
                            sample_means = np.array(sample_means)
                            
                            # Add histogram
                            row = i // 2 + 1
                            col = i % 2 + 1
                            fig.add_trace(go.Histogram(x=sample_means, nbinsx=30, name=f'n={sample_size}'),
                                        row=row, col=col)
                            
                            # Test normality of sample means
                            _, p_value = normaltest(sample_means)
                            
                            clt_results.append({
                                'Sample Size': sample_size,
                                'Mean of Sample Means': f"{np.mean(sample_means):.3f}",
                                'Std of Sample Means': f"{np.std(sample_means):.3f}",
                                'Theoretical Std': f"{np.std(col_data)/np.sqrt(sample_size):.3f}",
                                'Normality p-value': f"{p_value:.4f}",
                                'Normal': "Yes" if p_value > 0.05 else "No"
                            })
                        
                        fig.update_layout(height=600, title=f"Central Limit Theorem - {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # CLT results table
                        clt_df = pd.DataFrame(clt_results)
                        st.dataframe(clt_df, use_container_width=True)
                        
                        st.info("📊 As sample size increases, the distribution of sample means approaches normality (CLT)")

    def randomness_entropy_analysis(self):
        """Analyze randomness and entropy in the data"""
        st.subheader("🎲 Randomness & Entropy Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"🎲 {name.title()} Randomness Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                # Entropy analysis for categorical variables
                if len(categorical_cols) > 0:
                    st.write("**📊 Entropy Analysis (Categorical Variables)**")
                    
                    entropy_results = []
                    for col in categorical_cols:
                        value_counts = df[col].value_counts()
                        probabilities = value_counts / value_counts.sum()
                        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon
                        max_entropy = np.log2(len(probabilities))
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        
                        entropy_results.append({
                            'Variable': col,
                            'Unique Values': len(probabilities),
                            'Entropy': f"{entropy:.3f}",
                            'Max Possible Entropy': f"{max_entropy:.3f}",
                            'Normalized Entropy': f"{normalized_entropy:.3f}",
                            'Randomness Level': self._assess_randomness(normalized_entropy)
                        })
                    
                    entropy_df = pd.DataFrame(entropy_results)
                    st.dataframe(entropy_df, use_container_width=True)
                
                # Runs test for randomness (numerical variables)
                if len(numerical_cols) > 0:
                    st.write("**🔄 Runs Test for Randomness (Numerical Variables)**")
                    
                    randomness_results = []
                    for col in numerical_cols:
                        col_data = df[col].dropna()
                        if len(col_data) > 10:
                            median_val = col_data.median()
                            binary_sequence = (col_data > median_val).astype(int)
                            
                            # Count runs
                            runs = 1
                            for i in range(1, len(binary_sequence)):
                                if binary_sequence.iloc[i] != binary_sequence.iloc[i-1]:
                                    runs += 1
                            
                            # Expected runs and variance for random sequence
                            n1 = (binary_sequence == 1).sum()
                            n2 = (binary_sequence == 0).sum()
                            n = n1 + n2
                            
                            expected_runs = (2 * n1 * n2) / n + 1
                            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
                            
                            # Z-score
                            if variance_runs > 0:
                                z_score = (runs - expected_runs) / np.sqrt(variance_runs)
                                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                            else:
                                z_score, p_value = np.nan, np.nan
                            
                            randomness_results.append({
                                'Variable': col,
                                'Observed Runs': runs,
                                'Expected Runs': f"{expected_runs:.2f}",
                                'Z-score': f"{z_score:.3f}" if not np.isnan(z_score) else "N/A",
                                'P-value': f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
                                'Random': "Yes" if not np.isnan(p_value) and p_value > 0.05 else "No"
                            })
                    
                    if randomness_results:
                        rand_df = pd.DataFrame(randomness_results)
                        st.dataframe(rand_df, use_container_width=True)

    def _assess_randomness(self, normalized_entropy):
        """Assess randomness level based on normalized entropy"""
        if normalized_entropy > 0.9:
            return "High Randomness"
        elif normalized_entropy > 0.7:
            return "Moderate Randomness"
        elif normalized_entropy > 0.4:
            return "Low Randomness"
        else:
            return "Very Low Randomness"

    def distance_similarity_metrics(self):
        """Calculate various distance and similarity metrics"""
        st.subheader("📏 Distance & Similarity Metrics")
        
        for name, df in self.datasets.items():
            with st.expander(f"📏 {name.title()} Distance Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 1:
                    # Select subset of data for demonstration
                    sample_size = min(100, len(df))
                    df_sample = df[numerical_cols].sample(n=sample_size).fillna(0)
                    
                    # Calculate different distance matrices
                    from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
                    
                    distances = {
                        'Euclidean': euclidean_distances(df_sample),
                        'Manhattan': manhattan_distances(df_sample),
                        'Cosine': cosine_distances(df_sample)
                    }
                    
                    # Visualize distance matrices
                    fig = make_subplots(rows=1, cols=3,
                                      subplot_titles=['Euclidean Distance', 'Manhattan Distance', 'Cosine Distance'])
                    
                    for i, (dist_name, dist_matrix) in enumerate(distances.items()):
                        fig.add_trace(go.Heatmap(z=dist_matrix, showscale=(i==0), 
                                               colorscale='Viridis', name=dist_name),
                                    row=1, col=i+1)
                    
                    fig.update_layout(height=400, title="Distance Matrices Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Distance statistics
                    distance_stats = {}
                    for dist_name, dist_matrix in distances.items():
                        # Extract upper triangle (excluding diagonal)
                        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
                        
                        distance_stats[dist_name] = {
                            'Mean Distance': np.mean(upper_triangle),
                            'Std Distance': np.std(upper_triangle),
                            'Min Distance': np.min(upper_triangle),
                            'Max Distance': np.max(upper_triangle)
                        }
                    
                    # Display statistics
                    stats_df = pd.DataFrame(distance_stats).T
                    st.dataframe(stats_df, use_container_width=True)

    def multicollinearity_analysis(self):
        """Analyze multicollinearity among variables"""
        st.subheader("🌐 Multicollinearity Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"🌐 {name.title()} Multicollinearity", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 2:
                    # Calculate VIF (Variance Inflation Factor)
                    df_clean = df[numerical_cols].dropna()
                    
                    if len(df_clean) > len(numerical_cols):
                        vif_data = []
                        
                        for i, col in enumerate(numerical_cols):
                            # Calculate R-squared for regression of col on all other variables
                            X = df_clean.drop(columns=[col])
                            y = df_clean[col]
                            
                            if len(X.columns) > 0:
                                from sklearn.linear_model import LinearRegression
                                from sklearn.metrics import r2_score
                                
                                model = LinearRegression()
                                try:
                                    model.fit(X, y)
                                    y_pred = model.predict(X)
                                    r_squared = r2_score(y, y_pred)
                                    
                                    # VIF = 1 / (1 - R²)
                                    vif = 1 / (1 - r_squared) if r_squared < 0.99 else float('inf')
                                    
                                    vif_data.append({
                                        'Variable': col,
                                        'VIF': f"{vif:.3f}" if vif != float('inf') else "∞",
                                        'R-squared': f"{r_squared:.3f}",
                                        'Multicollinearity': self._assess_multicollinearity(vif)
                                    })
                                except:
                                    vif_data.append({
                                        'Variable': col,
                                        'VIF': "Error",
                                        'R-squared': "Error",
                                        'Multicollinearity': "Cannot Calculate"
                                    })
                        
                        if vif_data:
                            vif_df = pd.DataFrame(vif_data)
                            st.dataframe(vif_df, use_container_width=True)
                            
                            st.info("""
                            **VIF Interpretation:**
                            - VIF = 1: No multicollinearity
                            - 1 < VIF < 5: Moderate multicollinearity
                            - VIF ≥ 5: High multicollinearity
                            - VIF ≥ 10: Very high multicollinearity
                            """)

    def _assess_multicollinearity(self, vif):
        """Assess multicollinearity level based on VIF"""
        if vif == float('inf'):
            return "Perfect Multicollinearity"
        elif vif >= 10:
            return "Very High"
        elif vif >= 5:
            return "High"
        elif vif > 1:
            return "Moderate"
        else:
            return "Low"

    def principal_component_analysis(self):
        """Perform Principal Component Analysis"""
        st.subheader("🔀 Principal Component Analysis")
        
        for name, df in self.datasets.items():
            with st.expander(f"🔀 {name.title()} PCA Analysis", expanded=True):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numerical_cols) > 2:
                    # Prepare data
                    df_clean = df[numerical_cols].dropna()
                    
                    if len(df_clean) > len(numerical_cols):
                        # Standardize the data
                        scaler = StandardScaler()
                        df_scaled = scaler.fit_transform(df_clean)
                        
                        # Perform PCA
                        n_components = min(len(numerical_cols), 10)  # Limit to 10 components
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(df_scaled)
                        
                        # Explained variance
                        explained_variance = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        # Visualization
                        fig = make_subplots(rows=2, cols=2,
                                          subplot_titles=['Explained Variance', 'Cumulative Variance',
                                                        'PC1 vs PC2', 'Component Loadings'])
                        
                        # Explained variance
                        fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(explained_variance))],
                                           y=explained_variance, name='Explained Variance'),
                                    row=1, col=1)
                        
                        # Cumulative variance
                        fig.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                                               y=cumulative_variance, name='Cumulative Variance'),
                                    row=1, col=2)
                        
                        # PC1 vs PC2 scatter
                        fig.add_trace(go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1],
                                               mode='markers', name='Data Points'),
                                    row=2, col=1)
                        
                        # Component loadings (first 2 components)
                        loadings = pca.components_[:2].T
                        for i, var in enumerate(numerical_cols):
                            fig.add_trace(go.Scatter(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]],
                                                   mode='lines+text', text=[None, var],
                                                   name=f'Loading {var}', showlegend=False),
                                        row=2, col=2)
                        
                        fig.update_layout(height=600, title=f"Principal Component Analysis - {name.title()}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # PCA summary
                        pca_summary = {
                            'Total Components': n_components,
                            'PC1 Variance Explained': f"{explained_variance[0]*100:.2f}%",
                            'PC2 Variance Explained': f"{explained_variance[1]*100:.2f}%",
                            'First 2 PCs Total': f"{(explained_variance[0] + explained_variance[1])*100:.2f}%",
                            'Components for 80% Variance': np.argmax(cumulative_variance >= 0.8) + 1,
                            'Components for 95% Variance': np.argmax(cumulative_variance >= 0.95) + 1
                        }
                        
                        st.json(pca_summary)
                        
                        # Component loadings table
                        loadings_df = pd.DataFrame(
                            pca.components_[:5].T,  # First 5 components
                            columns=[f'PC{i+1}' for i in range(min(5, n_components))],
                            index=numerical_cols
                        )
                        st.write("**Component Loadings (First 5 Components):**")
                        st.dataframe(loadings_df, use_container_width=True)
