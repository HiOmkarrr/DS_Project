"""
Experiment 5: Explainable AI & Model Interpretability
SHAP, LIME, and advanced interpretability techniques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Experiment5:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        self.data_path = r"C:\Users\Omkar\SEM_7\DS_Project\datasets\DS-2-8-25"
        
    def render_explainable_ai_dashboard(self):
        """Main Explainable AI dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>🔍 Explainable AI & Model Interpretability</h2>
            <p>Understanding model decisions through advanced interpretability techniques</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if models are available
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models in Experiment 4 first")
            return
        
        # XAI Options
        xai_options = [
            "🎯 Model Interpretability Overview",
            "🌟 SHAP Analysis (Global & Local)",
            "🔬 LIME Explanations",
            "📊 Feature Importance Deep Dive",
            "🎭 Partial Dependence Plots",
            "🔗 Feature Interaction Analysis", 
            "⚖️ Model Fairness Assessment",
            "📋 Interpretability Report Generator",
            "🎨 Custom Explanation Dashboard"
        ]
        
        selected_activity = st.selectbox(
            "Select XAI Analysis:",
            xai_options
        )
        
        # Route to appropriate activity
        if selected_activity == "🎯 Model Interpretability Overview":
            self.interpretability_overview()
        elif selected_activity == "🌟 SHAP Analysis (Global & Local)":
            self.shap_analysis()
        elif selected_activity == "🔬 LIME Explanations":
            self.lime_explanations()
        elif selected_activity == "📊 Feature Importance Deep Dive":
            self.feature_importance_deep_dive()
        elif selected_activity == "🎭 Partial Dependence Plots":
            self.partial_dependence_plots()
        elif selected_activity == "🔗 Feature Interaction Analysis":
            self.feature_interaction_analysis()
        elif selected_activity == "⚖️ Model Fairness Assessment":
            self.model_fairness_assessment()
        elif selected_activity == "📋 Interpretability Report Generator":
            self.interpretability_report()
        elif selected_activity == "🎨 Custom Explanation Dashboard":
            self.custom_explanation_dashboard()

    def interpretability_overview(self):
        """Overview of model interpretability concepts"""
        st.subheader("🎯 Model Interpretability Overview")
        
        # Interpretability concepts
        st.markdown("""
        ### 🧠 Understanding Model Interpretability
        
        **Global Interpretability:** Understanding the overall behavior of the model
        - Which features are most important across all predictions?
        - How do features interact with each other?
        - What patterns does the model learn?
        
        **Local Interpretability:** Understanding individual predictions
        - Why did the model make this specific prediction?
        - Which features contributed most to this decision?
        - How would changing feature values affect the prediction?
        """)
        
        # Available models
        models = st.session_state.ml_data['trained_models']
        
        st.subheader("📊 Available Models for Analysis")
        
        model_info = []
        for model_name, model in models.items():
            model_type = type(model).__name__
            has_feature_importance = hasattr(model, 'feature_importances_')
            has_predict_proba = hasattr(model, 'predict_proba')
            
            model_info.append({
                'Model': model_name,
                'Type': model_type,
                'Feature Importance': '✅' if has_feature_importance else '❌',
                'Probability Support': '✅' if has_predict_proba else '❌',
                'Interpretability': self._get_interpretability_score(model_type)
            })
        
        model_df = pd.DataFrame(model_info)
        st.dataframe(model_df, use_container_width=True)
        
        # Interpretability techniques
        st.subheader("🔬 Available Interpretability Techniques")
        
        techniques = {
            'SHAP (SHapley Additive exPlanations)': {
                'Description': 'Unified framework for interpreting predictions based on game theory',
                'Best For': 'Most model types, both global and local explanations',
                'Pros': 'Mathematically grounded, consistent, efficient',
                'Cons': 'Can be computationally expensive for large datasets'
            },
            'LIME (Local Interpretable Model-agnostic Explanations)': {
                'Description': 'Explains individual predictions by learning local interpretable models',
                'Best For': 'Any black-box model, great for local explanations',
                'Pros': 'Model-agnostic, intuitive explanations',
                'Cons': 'Only local explanations, can be unstable'
            },
            'Feature Importance': {
                'Description': 'Built-in importance scores from tree-based models',
                'Best For': 'Tree-based models (Random Forest, XGBoost)',
                'Pros': 'Fast, built-in, global view',
                'Cons': 'Limited to certain model types, no local explanations'
            },
            'Partial Dependence Plots': {
                'Description': 'Shows the marginal effect of features on predictions',
                'Best For': 'Understanding feature-target relationships',
                'Pros': 'Model-agnostic, shows non-linear relationships',
                'Cons': 'Assumes feature independence'
            }
        }
        
        for technique, info in techniques.items():
            with st.expander(f"📖 {technique}"):
                st.write(f"**Description:** {info['Description']}")
                st.write(f"**Best For:** {info['Best For']}")
                st.write(f"**Pros:** {info['Pros']}")
                st.write(f"**Cons:** {info['Cons']}")

    def _get_interpretability_score(self, model_type):
        """Get interpretability score for model type"""
        interpretability_map = {
            'LogisticRegression': 'High',
            'DecisionTreeClassifier': 'High',
            'RandomForestClassifier': 'Medium',
            'GradientBoostingClassifier': 'Medium',
            'SVC': 'Low',
            'KNeighborsClassifier': 'Medium',
            'GaussianNB': 'Medium'
        }
        return interpretability_map.get(model_type, 'Unknown')

    def shap_analysis(self):
        """SHAP analysis for model explanations"""
        st.subheader("🌟 SHAP Analysis (Global & Local)")
        
        try:
            import shap
        except ImportError:
            st.error("❌ SHAP not installed. Please install with: pip install shap")
            st.code("pip install shap", language='bash')
            return
        
        models = st.session_state.ml_data['trained_models']
        X_test = st.session_state.ml_data['X_test']
        
        # Model selection
        selected_model = st.selectbox("Select Model for SHAP Analysis:", list(models.keys()))
        model = models[selected_model]
        
        # SHAP analysis type
        analysis_type = st.radio(
            "Select SHAP Analysis Type:",
            ["🌍 Global Explanations", "🎯 Local Explanations", "📊 Summary Plots"]
        )
        
        if st.button("🚀 Generate SHAP Explanations"):
            try:
                # Create SHAP explainer
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model) if 'Tree' in type(model).__name__ or 'Forest' in type(model).__name__ else shap.Explainer(model)
                else:
                    explainer = shap.Explainer(model)
                
                # Calculate SHAP values (sample for performance)
                sample_size = min(100, len(X_test))
                X_sample = X_test.sample(sample_size)
                shap_values = explainer(X_sample)
                
                if analysis_type == "🌍 Global Explanations":
                    self._global_shap_analysis(shap_values, X_sample)
                elif analysis_type == "🎯 Local Explanations":
                    self._local_shap_analysis(shap_values, X_sample)
                elif analysis_type == "📊 Summary Plots":
                    self._shap_summary_plots(shap_values, X_sample)
                    
            except Exception as e:
                st.error(f"❌ SHAP analysis failed: {str(e)}")
                st.info("💡 Trying alternative approach with sample data...")
                self._alternative_shap_analysis(model, X_test)

    def _global_shap_analysis(self, shap_values, X_sample):
        """Global SHAP analysis"""
        st.subheader("🌍 Global SHAP Analysis")
        
        # Feature importance from SHAP
        feature_importance = np.abs(shap_values.values).mean(0)
        feature_names = X_sample.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Importance': feature_importance
        }).sort_values('SHAP Importance', ascending=False)
        
        # Plot global importance
        fig = px.bar(importance_df.head(15), 
                    x='SHAP Importance', y='Feature',
                    orientation='h',
                    title="Global Feature Importance (SHAP)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(importance_df, use_container_width=True)

    def _local_shap_analysis(self, shap_values, X_sample):
        """Local SHAP analysis for individual predictions"""
        st.subheader("🎯 Local SHAP Analysis")
        
        # Select instance for explanation
        instance_idx = st.selectbox("Select Instance to Explain:", range(len(X_sample)))
        
        # Get SHAP values for selected instance
        instance_shap = shap_values.values[instance_idx]
        instance_data = X_sample.iloc[instance_idx]
        
        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Feature Value': instance_data.values,
            'SHAP Value': instance_shap
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Waterfall plot data
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(explanation_df),
            y=explanation_df['Feature'],
            x=explanation_df['SHAP Value'],
            text=explanation_df['Feature Value'],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(title=f"SHAP Explanation for Instance {instance_idx}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed explanation table
        st.subheader("📋 Detailed Explanation")
        st.dataframe(explanation_df, use_container_width=True)

    def _shap_summary_plots(self, shap_values, X_sample):
        """SHAP summary plots"""
        st.subheader("📊 SHAP Summary Plots")
        
        # Feature importance vs impact
        feature_importance = np.abs(shap_values.values).mean(0)
        
        # Create summary dataframe
        summary_data = []
        for i, feature in enumerate(X_sample.columns):
            for j in range(len(shap_values.values)):
                summary_data.append({
                    'Feature': feature,
                    'SHAP Value': shap_values.values[j, i],
                    'Feature Value': X_sample.iloc[j, i],
                    'Abs SHAP': abs(shap_values.values[j, i])
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # SHAP bee swarm plot
        fig = px.scatter(summary_df, 
                        x='SHAP Value', y='Feature',
                        color='Feature Value',
                        title="SHAP Summary Plot",
                        hover_data=['Feature Value'])
        st.plotly_chart(fig, use_container_width=True)

    def _alternative_shap_analysis(self, model, X_test):
        """Alternative SHAP analysis approach"""
        st.info("🔄 Using simplified SHAP analysis...")
        
        # Simple feature importance visualization
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X_test.columns
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(15),
                        x='Importance', y='Feature',
                        orientation='h',
                        title="Feature Importance (Model-based)")
            st.plotly_chart(fig, use_container_width=True)

    def lime_explanations(self):
        """LIME explanations for individual predictions"""
        st.subheader("🔬 LIME Explanations")
        
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            st.error("❌ LIME not installed. Please install with: pip install lime")
            st.code("pip install lime", language='bash')
            return
        
        models = st.session_state.ml_data['trained_models']
        X_train = st.session_state.ml_data['X_train']
        X_test = st.session_state.ml_data['X_test']
        
        # Model selection
        selected_model = st.selectbox("Select Model for LIME Analysis:", list(models.keys()))
        model = models[selected_model]
        
        # Instance selection
        instance_idx = st.selectbox("Select Instance to Explain:", range(min(50, len(X_test))))
        
        if st.button("🚀 Generate LIME Explanation"):
            try:
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=X_train.columns,
                    mode='classification' if st.session_state.ml_data['problem_type'] == 'classification' else 'regression'
                )
                
                # Get explanation for selected instance
                instance = X_test.iloc[instance_idx].values
                
                if hasattr(model, 'predict_proba'):
                    explanation = explainer.explain_instance(instance, model.predict_proba)
                else:
                    explanation = explainer.explain_instance(instance, model.predict)
                
                # Extract explanation data
                exp_data = explanation.as_list()
                
                # Create visualization
                features = [item[0] for item in exp_data]
                weights = [item[1] for item in exp_data]
                
                colors = ['red' if w < 0 else 'green' for w in weights]
                
                fig = go.Figure(go.Bar(
                    y=features,
                    x=weights,
                    orientation='h',
                    marker_color=colors,
                    title=f"LIME Explanation for Instance {instance_idx}"
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation table
                exp_df = pd.DataFrame(exp_data, columns=['Feature', 'Weight'])
                st.dataframe(exp_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ LIME analysis failed: {str(e)}")

    def feature_importance_deep_dive(self):
        """Deep dive into feature importance"""
        st.subheader("📊 Feature Importance Deep Dive")
        
        models = st.session_state.ml_data['trained_models']
        X_train = st.session_state.ml_data['X_train']
        
        # Collect feature importance from all models
        importance_data = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
        
        if not importance_data:
            st.warning("⚠️ No models with feature importance available")
            return
        
        # Create comparison dataframe
        importance_df = pd.DataFrame(importance_data, index=X_train.columns)
        
        # Feature importance comparison
        st.subheader("🔄 Feature Importance Comparison Across Models")
        
        # Heatmap
        fig = px.imshow(importance_df.T, 
                       title="Feature Importance Heatmap",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features analysis
        st.subheader("🏆 Top Features Analysis")
        
        # Calculate average importance
        importance_df['Average'] = importance_df.mean(axis=1)
        importance_df['Std'] = importance_df.std(axis=1)
        
        top_features = importance_df.sort_values('Average', ascending=False).head(15)
        
        fig = px.bar(
            x=top_features['Average'],
            y=top_features.index,
            error_x=top_features['Std'],
            orientation='h',
            title="Top 15 Features (Average Importance)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature stability analysis
        st.subheader("📈 Feature Importance Stability")
        
        stability_df = importance_df.copy()
        stability_df['Coefficient_of_Variation'] = stability_df['Std'] / stability_df['Average']
        stability_df = stability_df.sort_values('Coefficient_of_Variation')
        
        fig = px.scatter(
            x=stability_df['Average'],
            y=stability_df['Coefficient_of_Variation'],
            hover_name=stability_df.index,
            title="Feature Importance Stability (Lower CV = More Stable)"
        )
        st.plotly_chart(fig, use_container_width=True)

    def partial_dependence_plots(self):
        """Generate partial dependence plots"""
        st.subheader("🎭 Partial Dependence Plots")
        
        try:
            from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        except ImportError:
            st.error("❌ Partial dependence plots require scikit-learn >= 0.22")
            return
        
        models = st.session_state.ml_data['trained_models']
        X_test = st.session_state.ml_data['X_test']
        
        # Model selection
        selected_model = st.selectbox("Select Model for PDP Analysis:", list(models.keys()))
        model = models[selected_model]
        
        # Feature selection
        numerical_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Select Features for PDP:",
            numerical_features,
            default=numerical_features[:3]
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature")
            return
        
        if st.button("🚀 Generate Partial Dependence Plots"):
            try:
                for feature in selected_features:
                    feature_idx = X_test.columns.get_loc(feature)
                    
                    # Calculate partial dependence
                    pd_result = partial_dependence(
                        model, X_test, features=[feature_idx], kind="average"
                    )
                    
                    # Create plot
                    fig = px.line(
                        x=pd_result['grid_values'][0],
                        y=pd_result['average'][0],
                        title=f"Partial Dependence Plot - {feature}"
                    )
                    fig.update_xaxis(title=feature)
                    fig.update_yaxis(title="Partial Dependence")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ PDP generation failed: {str(e)}")

    def feature_interaction_analysis(self):
        """Analyze feature interactions"""
        st.subheader("🔗 Feature Interaction Analysis")
        
        X_test = st.session_state.ml_data['X_test']
        
        # Calculate correlation matrix
        correlation_matrix = X_test.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if strong_correlations:
            st.subheader("🔗 Strong Feature Correlations")
            corr_df = pd.DataFrame(strong_correlations)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True)
            
            # Visualize top correlations
            top_corr = corr_df.head(10)
            
            fig = px.bar(
                top_corr,
                x='Correlation',
                y=[f"{row['Feature 1']} vs {row['Feature 2']}" for _, row in top_corr.iterrows()],
                orientation='h',
                title="Top 10 Feature Correlations"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No strong feature correlations found (|r| > 0.5)")
        
        # Correlation heatmap
        st.subheader("🌡️ Correlation Heatmap")
        fig = px.imshow(correlation_matrix, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    def model_fairness_assessment(self):
        """Assess model fairness and bias"""
        st.subheader("⚖️ Model Fairness Assessment")
        
        st.markdown("""
        ### 🎯 Fairness in Machine Learning
        
        Model fairness ensures that ML systems don't discriminate against protected groups.
        Key fairness metrics include:
        
        - **Demographic Parity:** Equal positive prediction rates across groups
        - **Equalized Odds:** Equal true positive and false positive rates across groups  
        - **Equal Opportunity:** Equal true positive rates across groups
        """)
        
        # Check if sensitive attributes exist
        X_test = st.session_state.ml_data['X_test']
        
        # Look for potential sensitive attributes
        potential_sensitive = []
        for col in X_test.columns:
            if any(keyword in col.lower() for keyword in ['gender', 'age', 'race', 'ethnic', 'income']):
                potential_sensitive.append(col)
        
        if potential_sensitive:
            st.subheader("🔍 Potential Sensitive Attributes Found")
            st.write(f"Detected columns: {', '.join(potential_sensitive)}")
            
            selected_sensitive = st.selectbox("Select Sensitive Attribute:", potential_sensitive)
            
            if st.button("🚀 Analyze Fairness"):
                self._analyze_fairness(selected_sensitive)
        else:
            st.info("ℹ️ No obvious sensitive attributes detected in the dataset")
            st.write("Consider adding demographic information for comprehensive fairness analysis")

    def _analyze_fairness(self, sensitive_attr):
        """Analyze fairness for a sensitive attribute"""
        X_test = st.session_state.ml_data['X_test']
        y_test = st.session_state.ml_data['y_test']
        
        # Get predictions from best model
        models = st.session_state.ml_data['trained_models']
        model_name = list(models.keys())[0]  # Use first model
        model = models[model_name]
        
        y_pred = model.predict(X_test)
        
        # Group by sensitive attribute
        groups = X_test[sensitive_attr].unique()
        
        fairness_metrics = []
        
        for group in groups:
            group_mask = X_test[sensitive_attr] == group
            group_y_true = y_test[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate metrics
            if st.session_state.ml_data['problem_type'] == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                accuracy = accuracy_score(group_y_true, group_y_pred)
                precision = precision_score(group_y_true, group_y_pred, average='weighted')
                recall = recall_score(group_y_true, group_y_pred, average='weighted')
                positive_rate = np.mean(group_y_pred)
                
                fairness_metrics.append({
                    'Group': group,
                    'Count': len(group_y_true),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Positive Rate': positive_rate
                })
        
        fairness_df = pd.DataFrame(fairness_metrics)
        
        st.subheader("📊 Fairness Metrics by Group")
        st.dataframe(fairness_df, use_container_width=True)
        
        # Visualize fairness metrics
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'Positive Rate']
        
        for metric in metrics_to_plot:
            fig = px.bar(fairness_df, x='Group', y=metric,
                        title=f"{metric} by {sensitive_attr}")
            st.plotly_chart(fig, use_container_width=True)

    def interpretability_report(self):
        """Generate comprehensive interpretability report"""
        st.subheader("📋 Interpretability Report Generator")
        
        models = st.session_state.ml_data['trained_models']
        
        st.markdown("""
        ### 📊 Model Interpretability Report
        
        This report provides a comprehensive overview of model interpretability across all trained models.
        """)
        
        # Model overview
        st.subheader("🤖 Model Overview")
        
        model_summary = []
        for model_name, model in models.items():
            model_type = type(model).__name__
            interpretability = self._get_interpretability_score(model_type)
            
            model_summary.append({
                'Model': model_name,
                'Type': model_type,
                'Interpretability Level': interpretability,
                'Feature Importance': '✅' if hasattr(model, 'feature_importances_') else '❌',
                'Probability Support': '✅' if hasattr(model, 'predict_proba') else '❌'
            })
        
        summary_df = pd.DataFrame(model_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Feature importance summary
        if any(hasattr(model, 'feature_importances_') for model in models.values()):
            st.subheader("🎯 Feature Importance Summary")
            
            # Collect feature importance
            importance_data = {}
            X_train = st.session_state.ml_data['X_train']
            
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data, index=X_train.columns)
                importance_df['Average'] = importance_df.mean(axis=1)
                top_features = importance_df.sort_values('Average', ascending=False).head(10)
                
                st.write("**Top 10 Most Important Features:**")
                st.dataframe(top_features, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Interpretability Recommendations")
        
        recommendations = [
            "🔍 Use SHAP for comprehensive global and local explanations",
            "📊 Generate partial dependence plots for key features",
            "⚖️ Assess model fairness across different groups",
            "🎯 Focus on the top 10 most important features for business insights",
            "🔄 Compare explanations across multiple models for robustness",
            "📈 Monitor feature importance stability over time"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

    def custom_explanation_dashboard(self):
        """Custom interactive explanation dashboard"""
        st.subheader("🎨 Custom Explanation Dashboard")
        
        models = st.session_state.ml_data['trained_models']
        X_test = st.session_state.ml_data['X_test']
        
        # Model selection
        selected_model = st.selectbox("Select Model:", list(models.keys()))
        model = models[selected_model]
        
        # Create custom input interface
        st.subheader("🎛️ Custom Input Interface")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Input Features:**")
            input_data = {}
            
            for i, feature in enumerate(X_test.columns[:5]):  # Show first 5 features
                if X_test[feature].dtype in ['int64', 'float64']:
                    min_val = float(X_test[feature].min())
                    max_val = float(X_test[feature].max())
                    mean_val = float(X_test[feature].mean())
                    
                    input_data[feature] = st.slider(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"custom_{feature}"
                    )
        
        with col2:
            st.write("**Prediction & Explanation:**")
            
            if st.button("🎯 Get Prediction & Explanation"):
                # Make prediction
                input_df = pd.DataFrame([input_data])
                
                # Add missing features with default values
                for feature in X_test.columns:
                    if feature not in input_data:
                        input_df[feature] = X_test[feature].mean()
                
                # Reorder columns to match training data
                input_df = input_df[X_test.columns]
                
                prediction = model.predict(input_df)[0]
                
                st.success(f"🎯 Prediction: {prediction}")
                
                # Simple feature contribution (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    # Calculate contribution as feature_value * importance
                    contributions = []
                    for feature in X_test.columns:
                        importance = model.feature_importances_[X_test.columns.get_loc(feature)]
                        value = input_df[feature].iloc[0]
                        contribution = importance * abs(value)
                        contributions.append({
                            'Feature': feature,
                            'Value': value,
                            'Importance': importance,
                            'Contribution': contribution
                        })
                    
                    contrib_df = pd.DataFrame(contributions)
                    contrib_df = contrib_df.sort_values('Contribution', ascending=False).head(10)
                    
                    st.write("**Top Contributing Features:**")
                    fig = px.bar(contrib_df, x='Contribution', y='Feature',
                                orientation='h', title="Feature Contributions")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(contrib_df, use_container_width=True)
                
                # Confidence/Probability display
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_df)[0]
                    
                    if len(probabilities) == 2:  # Binary classification
                        confidence = max(probabilities)
                        st.info(f"🎯 Prediction Confidence: {confidence:.2%}")
                    else:  # Multi-class
                        prob_df = pd.DataFrame({
                            'Class': range(len(probabilities)),
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        st.write("**Class Probabilities:**")
                        fig = px.bar(prob_df, x='Probability', y='Class',
                                    orientation='h', title="Class Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive feature exploration
        st.subheader("🔍 Interactive Feature Exploration")
        
        feature_to_explore = st.selectbox(
            "Select Feature to Explore:",
            X_test.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        if feature_to_explore:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature distribution
                fig = px.histogram(X_test, x=feature_to_explore,
                                 title=f"Distribution of {feature_to_explore}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature vs target (if available)
                if 'y_test' in st.session_state.ml_data:
                    y_test = st.session_state.ml_data['y_test']
                    
                    if st.session_state.ml_data['problem_type'] == 'classification':
                        fig = px.box(x=y_test, y=X_test[feature_to_explore],
                                   title=f"{feature_to_explore} by Target Class")
                    else:
                        fig = px.scatter(x=X_test[feature_to_explore], y=y_test,
                                       title=f"{feature_to_explore} vs Target")
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison dashboard
        st.subheader("🔄 Model Comparison Dashboard")
        
        if len(models) > 1:
            comparison_metrics = []
            
            for model_name, model in models.items():
                # Basic model info
                model_info = {
                    'Model': model_name,
                    'Type': type(model).__name__,
                    'Interpretability': self._get_interpretability_score(type(model).__name__)
                }
                
                # Add feature importance if available
                if hasattr(model, 'feature_importances_'):
                    # Get top 3 features
                    top_indices = np.argsort(model.feature_importances_)[-3:][::-1]
                    top_features = [X_test.columns[i] for i in top_indices]
                    model_info['Top_Features'] = ', '.join(top_features)
                else:
                    model_info['Top_Features'] = 'N/A'
                
                comparison_metrics.append(model_info)
            
            comparison_df = pd.DataFrame(comparison_metrics)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Feature importance correlation between models
            importance_data = {}
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_
            
            if len(importance_data) > 1:
                importance_df = pd.DataFrame(importance_data, index=X_test.columns)
                correlation_matrix = importance_df.corr()
                
                st.write("**Feature Importance Correlation Between Models:**")
                fig = px.imshow(correlation_matrix, 
                               title="Model Agreement on Feature Importance",
                               color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
        
        # Export explanations
        st.subheader("📤 Export Explanations")
        
        if st.button("📊 Generate Explanation Report"):
            self._generate_explanation_report(models, X_test)

    def _generate_explanation_report(self, models, X_test):
        """Generate and download explanation report"""
        try:
            # Create comprehensive report
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models_analyzed': list(models.keys()),
                'dataset_info': {
                    'n_samples': len(X_test),
                    'n_features': len(X_test.columns),
                    'feature_names': X_test.columns.tolist()
                }
            }
            
            # Add feature importance data
            feature_importance = {}
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    feature_importance[model_name] = {
                        feature: float(importance) 
                        for feature, importance in zip(X_test.columns, model.feature_importances_)
                    }
            
            report_data['feature_importance'] = feature_importance
            
            # Add model interpretability scores
            interpretability_scores = {}
            for model_name, model in models.items():
                interpretability_scores[model_name] = self._get_interpretability_score(type(model).__name__)
            
            report_data['interpretability_scores'] = interpretability_scores
            
            # Convert to JSON for download
            import json
            report_json = json.dumps(report_data, indent=2)
            
            st.download_button(
                label="📄 Download Explanation Report (JSON)",
                data=report_json,
                file_name=f"model_explanation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Explanation report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate report: {str(e)}")

    def _create_model_summary_table(self, models):
        """Create summary table of all models"""
        summary_data = []
        
        for model_name, model in models.items():
            model_type = type(model).__name__
            
            summary_data.append({
                'Model Name': model_name,
                'Model Type': model_type,
                'Interpretability': self._get_interpretability_score(model_type),
                'Feature Importance': '✅' if hasattr(model, 'feature_importances_') else '❌',
                'Probability Support': '✅' if hasattr(model, 'predict_proba') else '❌',
                'SHAP Compatible': '✅' if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier'] else '⚠️',
                'LIME Compatible': '✅'  # LIME is model-agnostic
            })
        
        return pd.DataFrame(summary_data)

    def _calculate_model_complexity(self, model):
        """Calculate model complexity score"""
        model_type = type(model).__name__
        
        complexity_scores = {
            'LogisticRegression': 1,
            'DecisionTreeClassifier': 2,
            'RandomForestClassifier': 3,
            'GradientBoostingClassifier': 4,
            'SVC': 4,
            'KNeighborsClassifier': 2,
            'GaussianNB': 1
        }
        
        return complexity_scores.get(model_type, 3)

    def _get_explanation_methods(self, model):
        """Get available explanation methods for a model"""
        model_type = type(model).__name__
        methods = []
        
        # Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            methods.append('Feature Importance')
        
        # Coefficients for linear models
        if hasattr(model, 'coef_'):
            methods.append('Coefficients')
        
        # SHAP compatibility
        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 
                         'DecisionTreeClassifier', 'LogisticRegression']:
            methods.append('SHAP')
        
        # LIME (always available)
        methods.append('LIME')
        
        # Partial dependence plots
        methods.append('Partial Dependence')
        
        return methods
