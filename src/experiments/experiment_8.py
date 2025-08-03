"""
Experiment 8: Advanced Analytics & Business Intelligence
Advanced analytics, forecasting, and business intelligence dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Experiment8:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        self.data_path = r"C:\Users\Omkar\SEM_7\DS_Project\datasets\DS-2-8-25"
        
    def render_advanced_analytics_dashboard(self):
        """Main Advanced Analytics dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>📊 Advanced Analytics & Business Intelligence</h2>
            <p>Comprehensive business analytics, forecasting, and intelligence dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced Analytics Options
        analytics_options = [
            "📈 Executive Dashboard & KPIs",
            "🔮 Sales Forecasting & Predictions",
            "👥 Customer Segmentation & Analysis",
            "💰 Revenue Analytics & Optimization",
            "📊 Market Basket Analysis",
            "🎯 Recommendation Engine",
            "📉 Churn Prediction & Analysis",
            "🌐 Geographic Analytics",
            "⚡ Real-time Analytics Dashboard"
        ]
        
        selected_activity = st.selectbox(
            "Select Advanced Analytics:",
            analytics_options
        )
        
        # Route to appropriate activity
        if selected_activity == "📈 Executive Dashboard & KPIs":
            self.executive_dashboard()
        elif selected_activity == "🔮 Sales Forecasting & Predictions":
            self.sales_forecasting()
        elif selected_activity == "👥 Customer Segmentation & Analysis":
            self.customer_segmentation()
        elif selected_activity == "💰 Revenue Analytics & Optimization":
            self.revenue_analytics()
        elif selected_activity == "📊 Market Basket Analysis":
            self.market_basket_analysis()
        elif selected_activity == "🎯 Recommendation Engine":
            self.recommendation_engine()
        elif selected_activity == "📉 Churn Prediction & Analysis":
            self.churn_analysis()
        elif selected_activity == "🌐 Geographic Analytics":
            self.geographic_analytics()
        elif selected_activity == "⚡ Real-time Analytics Dashboard":
            self.realtime_dashboard()

    def executive_dashboard(self):
        """Executive dashboard with key KPIs"""
        st.subheader("📈 Executive Dashboard & KPIs")
        
        # Load data for analysis
        df = self._load_sample_data()
        
        # Key Performance Indicators
        st.subheader("🎯 Key Performance Indicators")
        
        # Calculate KPIs
        total_revenue = np.random.randint(500000, 1000000)
        total_customers = np.random.randint(5000, 10000)
        avg_order_value = np.random.randint(50, 200)
        conversion_rate = np.random.uniform(2.5, 8.5)
        
        # Display KPIs in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Revenue",
                value=f"${total_revenue:,}",
                delta=f"{np.random.uniform(5, 15):.1f}%"
            )
        
        with col2:
            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}",
                delta=f"{np.random.uniform(2, 10):.1f}%"
            )
        
        with col3:
            st.metric(
                label="🛒 Avg Order Value",
                value=f"${avg_order_value}",
                delta=f"{np.random.uniform(-5, 10):.1f}%"
            )
        
        with col4:
            st.metric(
                label="📊 Conversion Rate",
                value=f"{conversion_rate:.1f}%",
                delta=f"{np.random.uniform(-1, 2):.1f}%"
            )
        
        # Revenue trend
        st.subheader("📈 Revenue Trend Analysis")
        
        # Generate sample revenue data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        revenue_data = []
        
        for date in dates:
            base_revenue = 2000 + np.random.normal(0, 300)
            # Add seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            # Add trend
            trend_factor = 1 + 0.001 * date.dayofyear
            
            daily_revenue = base_revenue * seasonal_factor * trend_factor
            revenue_data.append({
                'Date': date,
                'Revenue': max(0, daily_revenue),
                'Month': date.strftime('%B'),
                'Quarter': f"Q{date.quarter}"
            })
        
        revenue_df = pd.DataFrame(revenue_data)
        
        # Revenue trend chart
        fig = px.line(revenue_df, x='Date', y='Revenue',
                     title='Daily Revenue Trend')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly comparison
        monthly_revenue = revenue_df.groupby('Month')['Revenue'].sum().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(monthly_revenue, x='Month', y='Revenue',
                        title='Monthly Revenue Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            quarterly_revenue = revenue_df.groupby('Quarter')['Revenue'].sum().reset_index()
            fig = px.pie(quarterly_revenue, values='Revenue', names='Quarter',
                        title='Quarterly Revenue Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance alerts
        st.subheader("🚨 Performance Alerts")
        
        alerts = [
            {"type": "success", "message": "Revenue target achieved for Q4 2024"},
            {"type": "warning", "message": "Customer acquisition cost increased by 15%"},
            {"type": "info", "message": "New market segment showing 25% growth"},
            {"type": "error", "message": "Churn rate exceeds threshold in premium segment"}
        ]
        
        for alert in alerts:
            if alert["type"] == "success":
                st.success(f"✅ {alert['message']}")
            elif alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']}")
            elif alert["type"] == "info":
                st.info(f"ℹ️ {alert['message']}")
            elif alert["type"] == "error":
                st.error(f"❌ {alert['message']}")

    def sales_forecasting(self):
        """Sales forecasting and predictions"""
        st.subheader("🔮 Sales Forecasting & Predictions")
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.selectbox(
                "Forecast Horizon:",
                ["30 days", "60 days", "90 days", "6 months", "1 year"]
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence Level:",
                min_value=80, max_value=99, value=95
            )
        
        # Generate historical sales data
        historical_data = self._generate_sales_data()
        
        # Display historical trend
        st.subheader("📊 Historical Sales Trend")
        
        fig = px.line(historical_data, x='date', y='sales',
                     title='Historical Sales Data')
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Generate Forecast"):
            # Simple forecasting simulation
            forecast_data = self._generate_forecast(
                historical_data, forecast_horizon, confidence_level
            )
            
            # Combine historical and forecast data
            combined_data = pd.concat([
                historical_data.assign(type='Historical'),
                forecast_data.assign(type='Forecast')
            ])
            
            # Forecast visualization
            fig = px.line(combined_data, x='date', y='sales', color='type',
                         title=f'Sales Forecast - {forecast_horizon}')
            
            # Add confidence intervals if available
            if 'upper_bound' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            st.subheader("📈 Forecast Metrics")
            
            forecast_total = forecast_data['sales'].sum()
            historical_avg = historical_data['sales'].mean()
            growth_rate = (forecast_data['sales'].mean() / historical_avg - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Forecasted Total Sales", f"${forecast_total:,.0f}")
            
            with col2:
                st.metric("Expected Growth Rate", f"{growth_rate:.1f}%")
            
            with col3:
                st.metric("Forecast Accuracy", f"{confidence_level}%")

    def customer_segmentation(self):
        """Customer segmentation analysis"""
        st.subheader("👥 Customer Segmentation & Analysis")
        
        # Generate customer data
        customer_data = self._generate_customer_data()
        
        # Segmentation method selection
        segmentation_method = st.selectbox(
            "Select Segmentation Method:",
            ["RFM Analysis", "Behavioral Segmentation", "Demographic Segmentation", "Value-based Segmentation"]
        )
        
        if segmentation_method == "RFM Analysis":
            self._rfm_analysis(customer_data)
        elif segmentation_method == "Behavioral Segmentation":
            self._behavioral_segmentation(customer_data)
        elif segmentation_method == "Demographic Segmentation":
            self._demographic_segmentation(customer_data)
        elif segmentation_method == "Value-based Segmentation":
            self._value_based_segmentation(customer_data)

    def _rfm_analysis(self, customer_data):
        """RFM (Recency, Frequency, Monetary) Analysis"""
        st.subheader("📊 RFM Analysis")
        
        # Calculate RFM scores
        customer_data['recency_score'] = pd.qcut(customer_data['recency'], 5, labels=[1,2,3,4,5])
        customer_data['frequency_score'] = pd.qcut(customer_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        customer_data['monetary_score'] = pd.qcut(customer_data['monetary'], 5, labels=[1,2,3,4,5])
        
        # Create RFM segments
        customer_data['rfm_score'] = (
            customer_data['recency_score'].astype(str) + 
            customer_data['frequency_score'].astype(str) + 
            customer_data['monetary_score'].astype(str)
        )
        
        # Define segment names
        def get_segment(rfm_score):
            if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif rfm_score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Potential Loyalists'
            elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
                return 'New Customers'
            elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
                return 'Cannot Lose Them'
            elif rfm_score in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234']:
                return 'At Risk'
            elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
                return 'Champions'
            else:
                return 'Others'
        
        customer_data['segment'] = customer_data['rfm_score'].apply(get_segment)
        
        # Segment distribution
        segment_counts = customer_data['segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title='Customer Segment Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment characteristics
        segment_analysis = customer_data.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        segment_analysis.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Count']
        st.dataframe(segment_analysis, use_container_width=True)
        
        # RFM heatmap
        rfm_agg = customer_data.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        fig = px.imshow(rfm_agg.T, title="RFM Segment Characteristics Heatmap",
                       color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig, use_container_width=True)

    def revenue_analytics(self):
        """Revenue analytics and optimization"""
        st.subheader("💰 Revenue Analytics & Optimization")
        
        # Revenue analysis options
        analysis_type = st.selectbox(
            "Select Revenue Analysis:",
            ["Revenue Streams", "Product Performance", "Pricing Analysis", "Revenue Optimization"]
        )
        
        if analysis_type == "Revenue Streams":
            self._revenue_streams_analysis()
        elif analysis_type == "Product Performance":
            self._product_performance_analysis()
        elif analysis_type == "Pricing Analysis":
            self._pricing_analysis()
        elif analysis_type == "Revenue Optimization":
            self._revenue_optimization()

    def _revenue_streams_analysis(self):
        """Analyze different revenue streams"""
        st.subheader("💰 Revenue Streams Analysis")
        
        # Generate revenue stream data
        revenue_streams = {
            'Product Sales': np.random.randint(300000, 500000),
            'Subscriptions': np.random.randint(100000, 200000),
            'Services': np.random.randint(50000, 150000),
            'Partnerships': np.random.randint(20000, 80000),
            'Licensing': np.random.randint(10000, 50000)
        }
        
        # Revenue stream breakdown
        fig = px.pie(values=list(revenue_streams.values()), 
                    names=list(revenue_streams.keys()),
                    title='Revenue Stream Breakdown')
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue trend by stream
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        trend_data = []
        
        for stream, base_value in revenue_streams.items():
            for i, date in enumerate(dates):
                # Add some randomness and trend
                value = base_value * (1 + 0.02 * i) * np.random.uniform(0.8, 1.2)
                trend_data.append({
                    'Date': date,
                    'Revenue': value,
                    'Stream': stream
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig = px.line(trend_df, x='Date', y='Revenue', color='Stream',
                     title='Revenue Trends by Stream')
        st.plotly_chart(fig, use_container_width=True)
        
        # Stream performance metrics
        st.subheader("📊 Stream Performance Metrics")
        
        performance_data = []
        for stream, revenue in revenue_streams.items():
            growth_rate = np.random.uniform(-5, 25)
            margin = np.random.uniform(15, 45)
            
            performance_data.append({
                'Revenue Stream': stream,
                'Revenue': f"${revenue:,}",
                'Growth Rate': f"{growth_rate:.1f}%",
                'Profit Margin': f"{margin:.1f}%",
                'Contribution': f"{revenue/sum(revenue_streams.values())*100:.1f}%"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)

    def market_basket_analysis(self):
        """Market basket analysis"""
        st.subheader("📊 Market Basket Analysis")
        
        # Generate sample transaction data
        transaction_data = self._generate_transaction_data()
        
        st.write("**Sample Transaction Data:**")
        st.dataframe(transaction_data.head(10), use_container_width=True)
        
        # Frequent itemsets
        st.subheader("🛒 Frequent Itemsets")
        
        # Simple frequent items analysis
        item_counts = {}
        for items in transaction_data['items']:
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Top selling items
        top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        items_df = pd.DataFrame(top_items, columns=['Item', 'Frequency'])
        
        fig = px.bar(items_df, x='Frequency', y='Item', orientation='h',
                    title='Top 10 Most Frequently Bought Items')
        st.plotly_chart(fig, use_container_width=True)
        
        # Association rules simulation
        st.subheader("🔗 Association Rules")
        
        # Generate sample association rules
        rules = [
            {"Antecedent": "Jeans", "Consequent": "T-Shirt", "Support": 0.15, "Confidence": 0.75, "Lift": 2.1},
            {"Antecedent": "Sneakers", "Consequent": "Socks", "Support": 0.12, "Confidence": 0.68, "Lift": 1.8},
            {"Antecedent": "Dress", "Consequent": "Heels", "Support": 0.10, "Confidence": 0.82, "Lift": 2.5},
            {"Antecedent": "Jacket", "Consequent": "Jeans", "Support": 0.08, "Confidence": 0.71, "Lift": 1.9},
            {"Antecedent": "Shirt", "Consequent": "Tie", "Support": 0.06, "Confidence": 0.85, "Lift": 3.2}
        ]
        
        rules_df = pd.DataFrame(rules)
        st.dataframe(rules_df, use_container_width=True)
        
        # Visualize rules
        fig = px.scatter(rules_df, x='Support', y='Confidence', size='Lift',
                        hover_data=['Antecedent', 'Consequent'],
                        title='Association Rules Visualization')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Business Recommendations")
        
        recommendations = [
            "🎯 Bundle Jeans and T-Shirts for increased sales",
            "👟 Place Socks near Sneakers display for cross-selling",
            "💃 Create Dress and Heels combo offers",
            "🧥 Promote Jacket and Jeans packages during winter",
            "👔 Suggest Ties with Shirt purchases at checkout"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

    def recommendation_engine(self):
        """Product recommendation engine"""
        st.subheader("🎯 Recommendation Engine")
        
        # Recommendation type selection
        rec_type = st.selectbox(
            "Select Recommendation Type:",
            ["Content-Based", "Collaborative Filtering", "Hybrid Approach"]
        )
        
        # Generate sample product data
        products = self._generate_product_data()
        
        # Customer selection for recommendations
        customer_id = st.selectbox(
            "Select Customer for Recommendations:",
            [f"Customer_{i}" for i in range(1, 11)]
        )
        
        if st.button("🚀 Generate Recommendations"):
            if rec_type == "Content-Based":
                recommendations = self._content_based_recommendations(customer_id, products)
            elif rec_type == "Collaborative Filtering":
                recommendations = self._collaborative_filtering_recommendations(customer_id, products)
            else:
                recommendations = self._hybrid_recommendations(customer_id, products)
            
            # Display recommendations
            st.subheader(f"📋 Recommendations for {customer_id}")
            
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)
            
            # Recommendation scores visualization
            fig = px.bar(rec_df, x='Recommendation_Score', y='Product',
                        orientation='h', title='Recommendation Scores')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation explanation
            st.subheader("🔍 Recommendation Explanation")
            
            explanations = {
                "Content-Based": "Based on product features similar to customer's purchase history",
                "Collaborative Filtering": "Based on preferences of similar customers",
                "Hybrid Approach": "Combines content and collaborative filtering for better accuracy"
            }
            
            st.info(f"**{rec_type}:** {explanations[rec_type]}")

    def churn_analysis(self):
        """Customer churn prediction and analysis"""
        st.subheader("📉 Churn Prediction & Analysis")
        
        # Generate customer churn data
        churn_data = self._generate_churn_data()
        
        # Churn overview
        st.subheader("📊 Churn Overview")
        
        churn_rate = churn_data['churned'].mean() * 100
        total_customers = len(churn_data)
        churned_customers = churn_data['churned'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            st.metric("Churned Customers", f"{churned_customers:,}")
        
        with col3:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Churn by segments
        churn_by_segment = churn_data.groupby('segment')['churned'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_segment.columns = ['Segment', 'Total_Customers', 'Churned_Customers', 'Churn_Rate']
        churn_by_segment['Churn_Rate'] = churn_by_segment['Churn_Rate'] * 100
        
        fig = px.bar(churn_by_segment, x='Segment', y='Churn_Rate',
                    title='Churn Rate by Customer Segment')
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn factors analysis
        st.subheader("🔍 Churn Factors Analysis")
        
        # Correlation with churn
        numerical_cols = ['tenure', 'total_spent', 'last_purchase_days', 'avg_order_value']
        
        correlations = []
        for col in numerical_cols:
            corr = churn_data[col].corr(churn_data['churned'])
            correlations.append({'Factor': col, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations)
        
        fig = px.bar(corr_df, x='Factor', y='Correlation',
                    title='Correlation with Churn')
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn prediction for individual customer
        st.subheader("🎯 Individual Churn Prediction")
        
        customer_id = st.selectbox(
            "Select Customer for Churn Prediction:",
            churn_data['customer_id'].unique()[:20]
        )
        
        if st.button("🔮 Predict Churn Risk"):
            customer_info = churn_data[churn_data['customer_id'] == customer_id].iloc[0]
            
            # Simple churn probability calculation
            risk_factors = {
                'Low Tenure': customer_info['tenure'] < 6,
                'Low Spending': customer_info['total_spent'] < churn_data['total_spent'].median(),
                'Recent Inactivity': customer_info['last_purchase_days'] > 30,
                'Low Order Value': customer_info['avg_order_value'] < churn_data['avg_order_value'].median()
            }
            
            risk_score = sum(risk_factors.values()) / len(risk_factors)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Customer Profile")
                st.write(f"**Tenure:** {customer_info['tenure']} months")
                st.write(f"**Total Spent:** ${customer_info['total_spent']:,.2f}")
                st.write(f"**Last Purchase:** {customer_info['last_purchase_days']} days ago")
                st.write(f"**Avg Order Value:** ${customer_info['avg_order_value']:,.2f}")
                st.write(f"**Segment:** {customer_info['segment']}")
            
            with col2:
                st.subheader("⚠️ Churn Risk Assessment")
                
                if risk_score < 0.3:
                    st.success(f"🟢 Low Risk ({risk_score:.1%})")
                elif risk_score < 0.6:
                    st.warning(f"🟡 Medium Risk ({risk_score:.1%})")
                else:
                    st.error(f"🔴 High Risk ({risk_score:.1%})")
                
                st.write("**Risk Factors:**")
                for factor, present in risk_factors.items():
                    status = "❌" if present else "✅"
                    st.write(f"{status} {factor}")

    def geographic_analytics(self):
        """Geographic analytics and insights"""
        st.subheader("🌐 Geographic Analytics")
        
        # Generate geographic data
        geo_data = self._generate_geographic_data()
        
        # Geographic distribution
        st.subheader("📍 Sales by Region")
        
        region_sales = geo_data.groupby('region')['sales'].sum().reset_index()
        
        fig = px.bar(region_sales, x='region', y='sales',
                    title='Sales Distribution by Region')
        st.plotly_chart(fig, use_container_width=True)
        
        # City-wise analysis
        st.subheader("🏙️ Top Cities by Sales")
        
        city_sales = geo_data.groupby('city')['sales'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=city_sales.values, y=city_sales.index, orientation='h',
                    title='Top 10 Cities by Sales')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by region
        st.subheader("📊 Regional Performance Metrics")
        
        region_metrics = geo_data.groupby('region').agg({
            'sales': ['sum', 'mean', 'count'],
            'customers': 'sum'
        }).round(2)
        
        region_metrics.columns = ['Total_Sales', 'Avg_Sales', 'Transactions', 'Customers']
        region_metrics['Sales_per_Customer'] = (region_metrics['Total_Sales'] / region_metrics['Customers']).round(2)
        
        st.dataframe(region_metrics, use_container_width=True)
        
        # Geographic insights
        st.subheader("🎯 Geographic Insights")
        
        insights = [
            f"🏆 Top performing region: {region_sales.loc[region_sales['sales'].idxmax(), 'region']}",
            f"🎯 Highest sales per customer: {region_metrics['Sales_per_Customer'].idxmax()}",
            f"📈 Most transactions: {region_metrics['Transactions'].idxmax()}",
            f"👥 Largest customer base: {region_metrics['Customers'].idxmax()}"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")

    def realtime_dashboard(self):
        """Real-time analytics dashboard"""
        st.subheader("⚡ Real-time Analytics Dashboard")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh (5 seconds)")
        
        if auto_refresh:
            import time
            time.sleep(5)
            st.rerun()
        
        # Real-time metrics
        st.subheader("📊 Live Metrics")
        
        # Simulate real-time data
        current_time = datetime.now()
        
        # Live KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_sales = np.random.randint(1000, 5000)
            st.metric("Current Sales", f"${current_sales:,}", f"+{np.random.randint(5, 20)}%")
        
        with col2:
            active_users = np.random.randint(100, 500)
            st.metric("Active Users", f"{active_users:,}", f"+{np.random.randint(-10, 30)}")
        
        with col3:
            conversion_rate = np.random.uniform(2, 8)
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%", f"{np.random.uniform(-0.5, 1.5):.1f}%")
        
        with col4:
            avg_session = np.random.uniform(3, 12)
            st.metric("Avg Session (min)", f"{avg_session:.1f}", f"{np.random.uniform(-1, 2):.1f}")
        
        # Live activity feed
        st.subheader("📈 Live Activity Feed")
        
        activities = [
            "🛒 New order placed - $156.99",
            "👤 New user registered - Premium plan",
            "💳 Payment completed - Order #12345",
            "📧 Email campaign sent - 5,000 recipients",
            "🎯 Goal achieved - Daily sales target",
            "⭐ New review posted - 5 stars",
            "🔄 Product updated - Inventory sync",
            "📱 Mobile app download - iOS"
        ]
        
        # Display random activities
        for _ in range(5):
            activity = np.random.choice(activities)
            timestamp = current_time - timedelta(minutes=np.random.randint(1, 60))
            st.write(f"**{timestamp.strftime('%H:%M:%S')}** - {activity}")
        
        # Real-time charts
        st.subheader("📊 Live Charts")
        
        # Generate real-time data
        time_points = pd.date_range(
            start=current_time - timedelta(hours=24),
            end=current_time,
            freq='H'
        )
        
        realtime_data = pd.DataFrame({
            'timestamp': time_points,
            'sales': np.random.randint(500, 2000, len(time_points)),
            'visitors': np.random.randint(100, 800, len(time_points)),
            'orders': np.random.randint(20, 100, len(time_points))
        })
        
        # Sales trend
        fig = px.line(realtime_data, x='timestamp', y='sales',
                     title='Sales - Last 24 Hours')
        st.plotly_chart(fig, use_container_width=True)
        
        # Multiple metrics
        fig = px.line(realtime_data, x='timestamp', y=['visitors', 'orders'],
                     title='Visitors vs Orders - Last 24 Hours')
        st.plotly_chart(fig, use_container_width=True)

    # Helper methods for data generation
    def _load_sample_data(self):
        """Load sample data for analysis"""
        # This would load actual data in a real implementation
        return pd.DataFrame()

    def _generate_sales_data(self):
        """Generate sample sales data"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        sales_data = []
        
        for date in dates:
            base_sales = 1500 + np.random.normal(0, 200)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            trend_factor = 1 + 0.0005 * (date - dates[0]).days
            
            daily_sales = base_sales * seasonal_factor * trend_factor
            sales_data.append({
                'date': date,
                'sales': max(0, daily_sales)
            })
        
        return pd.DataFrame(sales_data)

    def _generate_forecast(self, historical_data, horizon, confidence):
        """Generate forecast data"""
        horizon_days = {
            "30 days": 30, "60 days": 60, "90 days": 90,
            "6 months": 180, "1 year": 365
        }
        
        days = horizon_days[horizon]
        last_date = historical_data['date'].max()
        
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Simple trend extrapolation with noise
        last_sales = historical_data['sales'].tail(30).mean()
        trend = np.random.uniform(0.001, 0.003)
        
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            forecast_value = last_sales * (1 + trend * i) * np.random.uniform(0.9, 1.1)
            uncertainty = forecast_value * 0.1
            
            forecast_data.append({
                'date': date,
                'sales': forecast_value,
                'upper_bound': forecast_value + uncertainty,
                'lower_bound': max(0, forecast_value - uncertainty)
            })
        
        return pd.DataFrame(forecast_data)

    def _generate_customer_data(self):
        """Generate sample customer data"""
        n_customers = 1000
        
        customer_data = pd.DataFrame({
            'customer_id': [f"CUST_{i:04d}" for i in range(n_customers)],
            'recency': np.random.randint(1, 365, n_customers),
            'frequency': np.random.randint(1, 50, n_customers),
            'monetary': np.random.uniform(10, 5000, n_customers),
            'age': np.random.randint(18, 70, n_customers),
            'tenure': np.random.randint(1, 60, n_customers)
        })
        
        return customer_data

    def _behavioral_segmentation(self, customer_data):
        """Behavioral segmentation analysis"""
        st.write("**Behavioral Segmentation** - Based on customer behavior patterns")
        
        # Simple behavioral segmentation
        conditions = [
            (customer_data['frequency'] >= 20) & (customer_data['monetary'] >= 1000),
            (customer_data['frequency'] >= 10) & (customer_data['monetary'] >= 500),
            (customer_data['frequency'] >= 5) & (customer_data['monetary'] >= 200),
            (customer_data['frequency'] < 5) | (customer_data['monetary'] < 200)
        ]
        
        choices = ['Heavy Users', 'Regular Users', 'Light Users', 'Occasional Users']
        customer_data['behavior_segment'] = np.select(conditions, choices)
        
        segment_counts = customer_data['behavior_segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title='Behavioral Segmentation')
        st.plotly_chart(fig, use_container_width=True)

    def _demographic_segmentation(self, customer_data):
        """Demographic segmentation analysis"""
        st.write("**Demographic Segmentation** - Based on customer demographics")
        
        # Age-based segmentation
        age_conditions = [
            customer_data['age'] <= 25,
            (customer_data['age'] > 25) & (customer_data['age'] <= 35),
            (customer_data['age'] > 35) & (customer_data['age'] <= 50),
            customer_data['age'] > 50
        ]
        
        age_choices = ['Gen Z', 'Millennials', 'Gen X', 'Baby Boomers']
        customer_data['age_segment'] = np.select(age_conditions, age_choices)
        
        age_counts = customer_data['age_segment'].value_counts()
        
        fig = px.pie(values=age_counts.values, names=age_counts.index,
                    title='Age-based Segmentation')
        st.plotly_chart(fig, use_container_width=True)

    def _value_based_segmentation(self, customer_data):
        """Value-based segmentation analysis"""
        st.write("**Value-based Segmentation** - Based on customer lifetime value")
        
        # Calculate CLV proxy
        customer_data['clv'] = customer_data['monetary'] * customer_data['frequency']
        
        # Value segments
        customer_data['value_segment'] = pd.qcut(
            customer_data['clv'], 
            q=4, 
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium Value']
        )
        
        value_counts = customer_data['value_segment'].value_counts()
        
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    title='Value-based Segmentation')
        st.plotly_chart(fig, use_container_width=True)

    def _product_performance_analysis(self):
        """Product performance analysis"""
        st.write("**Product Performance Analysis**")
        
        # Generate sample product performance data
        products = ['T-Shirt', 'Jeans', 'Sneakers', 'Dress', 'Jacket', 'Shoes', 'Accessories']
        
        performance_data = []
        for product in products:
            performance_data.append({
                'Product': product,
                'Revenue': np.random.randint(50000, 200000),
                'Units_Sold': np.random.randint(500, 2000),
                'Avg_Price': np.random.uniform(30, 150),
                'Profit_Margin': np.random.uniform(20, 60)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        fig = px.bar(performance_df, x='Product', y='Revenue',
                    title='Product Revenue Performance')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(performance_df, use_container_width=True)

    def _pricing_analysis(self):
        """Pricing analysis"""
        st.write("**Pricing Analysis**")
        
        # Price elasticity simulation
        prices = np.arange(20, 101, 5)
        demand = 1000 * np.exp(-0.02 * prices) + np.random.normal(0, 50, len(prices))
        revenue = prices * demand
        
        pricing_df = pd.DataFrame({
            'Price': prices,
            'Demand': demand,
            'Revenue': revenue
        })
        
        fig = px.line(pricing_df, x='Price', y=['Demand', 'Revenue'],
                     title='Price vs Demand and Revenue')
        st.plotly_chart(fig, use_container_width=True)
        
        optimal_price = pricing_df.loc[pricing_df['Revenue'].idxmax(), 'Price']
        st.success(f"🎯 Optimal Price Point: ${optimal_price}")

    def _revenue_optimization(self):
        """Revenue optimization suggestions"""
        st.write("**Revenue Optimization Recommendations**")
        
        optimization_strategies = [
            {"Strategy": "Dynamic Pricing", "Impact": "15-25%", "Effort": "High"},
            {"Strategy": "Bundle Products", "Impact": "10-20%", "Effort": "Medium"},
            {"Strategy": "Upselling", "Impact": "8-15%", "Effort": "Low"},
            {"Strategy": "Cross-selling", "Impact": "12-18%", "Effort": "Medium"},
            {"Strategy": "Loyalty Programs", "Impact": "5-12%", "Effort": "High"},
            {"Strategy": "Personalization", "Impact": "10-30%", "Effort": "High"}
        ]
        
        strategy_df = pd.DataFrame(optimization_strategies)
        st.dataframe(strategy_df, use_container_width=True)

    def _generate_transaction_data(self):
        """Generate sample transaction data"""
        products = ['T-Shirt', 'Jeans', 'Sneakers', 'Dress', 'Jacket', 'Shoes', 'Accessories', 'Socks', 'Hat', 'Belt']
        
        transactions = []
        for i in range(1000):
            # Random number of items per transaction
            n_items = np.random.randint(1, 6)
            items = np.random.choice(products, size=n_items, replace=False).tolist()
            
            transactions.append({
                'transaction_id': f"TXN_{i:04d}",
                'items': items,
                'total_amount': np.random.uniform(20, 500)
            })
        
        return pd.DataFrame(transactions)

    def _generate_product_data(self):
        """Generate sample product data"""
        categories = ['Clothing', 'Footwear', 'Accessories']
        brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D']
        
        products = []
        for i in range(50):
            products.append({
                'product_id': f"PROD_{i:03d}",
                'name': f"Product {i}",
                'category': np.random.choice(categories),
                'brand': np.random.choice(brands),
                'price': np.random.uniform(20, 200),
                'rating': np.random.uniform(3.0, 5.0),
                'popularity': np.random.randint(1, 100)
            })
        
        return pd.DataFrame(products)

    def _content_based_recommendations(self, customer_id, products):
        """Generate content-based recommendations"""
        # Simulate recommendations based on product features
        recommendations = products.sample(5).copy()
        recommendations['Recommendation_Score'] = np.random.uniform(0.6, 0.95, 5)
        recommendations['Reason'] = 'Similar to past purchases'
        
        return recommendations[['name', 'category', 'price', 'Recommendation_Score', 'Reason']]

    def _collaborative_filtering_recommendations(self, customer_id, products):
        """Generate collaborative filtering recommendations"""
        recommendations = products.sample(5).copy()
        recommendations['Recommendation_Score'] = np.random.uniform(0.5, 0.9, 5)
        recommendations['Reason'] = 'Customers like you also bought'
        
        return recommendations[['name', 'category', 'price', 'Recommendation_Score', 'Reason']]

    def _hybrid_recommendations(self, customer_id, products):
        """Generate hybrid recommendations"""
        recommendations = products.sample(5).copy()
        recommendations['Recommendation_Score'] = np.random.uniform(0.7, 0.98, 5)
        recommendations['Reason'] = 'Hybrid approach'
        
        return recommendations[['name', 'category', 'price', 'Recommendation_Score', 'Reason']]

    def _generate_churn_data(self):
        """Generate sample churn data"""
        n_customers = 1000
        
        churn_data = pd.DataFrame({
            'customer_id': [f"CUST_{i:04d}" for i in range(n_customers)],
            'tenure': np.random.randint(1, 60, n_customers),
            'total_spent': np.random.uniform(100, 5000, n_customers),
            'last_purchase_days': np.random.randint(1, 180, n_customers),
            'avg_order_value': np.random.uniform(20, 300, n_customers),
            'segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers),
            'churned': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
        })
        
        return churn_data

    def _generate_geographic_data(self):
        """Generate sample geographic data"""
        regions = ['North', 'South', 'East', 'West', 'Central']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        geo_data = []
        for _ in range(500):
            geo_data.append({
                'region': np.random.choice(regions),
                'city': np.random.choice(cities),
                'sales': np.random.uniform(1000, 10000),
                'customers': np.random.randint(10, 100)
            })
        
        return pd.DataFrame(geo_data)
