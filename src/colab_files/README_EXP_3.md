# 🔍 Experiment 3: Exploratory Data Analysis & Statistical Analysis - Concepts Explained

## 🎯 Overview
This README explains all the statistical and data analysis concepts used in Experiment 3 (DS_EXP_3.ipynb) in simple terms with practical examples.

---

## 📋 Table of Contents
1. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
2. [Class Balance Analysis](#-class-balance-analysis)
3. [Feature Distribution Analysis](#-feature-distribution-analysis)
4. [Correlation Analysis](#-correlation-analysis)
5. [Statistical Hypothesis Testing](#-statistical-hypothesis-testing)
6. [Distribution Fitting](#-distribution-fitting)
7. [Outlier Detection Methods](#-outlier-detection-methods)
8. [Statistical Tests Explained](#-statistical-tests-explained)

---

## 🔍 Exploratory Data Analysis (EDA)

### **What is EDA?**
EDA is like being a detective with your data - you explore, investigate, and discover patterns before building models.

### **Why is it important?**
- Understand your data's "personality"
- Find hidden patterns and relationships
- Spot problems and anomalies
- Make informed decisions about modeling

### **EDA Process:**
```
1. Look at basic info (shape, types, missing values)
2. Visualize distributions and patterns
3. Find relationships between variables
4. Test statistical assumptions
5. Document insights and findings
```

### **Real-world analogy:**
Like exploring a new city before planning your route - you need to know the layout, traffic patterns, and landmarks.

---

## ⚖️ Class Balance Analysis

### **What is class balance?**
Checking if you have roughly equal amounts of different categories in your data.

### **Example - Product Categories:**
```python
# Balanced dataset:
T-Shirts: 250 products (25%)
Jeans: 240 products (24%)
Dresses: 230 products (23%)
Shoes: 280 products (28%)
Total: 1000 products

# Imbalanced dataset:
T-Shirts: 800 products (80%)
Jeans: 100 products (10%)
Dresses: 50 products (5%)
Shoes: 50 products (5%)
Total: 1000 products
```

### **Why does balance matter?**
- **Machine Learning**: Models prefer balanced data
- **Statistical Analysis**: Unbalanced data can skew results
- **Business Insights**: May indicate business focus or data collection bias

### **How to check:**
1. **Count Plot**: Bar chart showing frequency of each category
2. **Pie Chart**: Shows proportions as slices
3. **Percentage Table**: Exact percentages for each category

### **What to do if imbalanced:**
- **Collect more data** for underrepresented categories
- **Use sampling techniques** (oversample minority, undersample majority)
- **Use specialized algorithms** that handle imbalance

---

## 📊 Feature Distribution Analysis

### **What is distribution analysis?**
Understanding how your data values are spread out - are they normal, skewed, or have multiple peaks?

### **Types of distributions:**

#### **1. Normal Distribution (Bell Curve)**
```
   Frequency
       ∧
      /|\
     / | \     ← Most values in the middle
    /  |  \
   /   |   \
  /    |    \
 /     |     \
/______|______\
     Values
```
**Example:** Heights of people, test scores

#### **2. Skewed Distribution**
```
Right-skewed (tail on right):
Frequency
   ∧
   ||\
   || \
   ||  \
   ||   \____
   ||        \___
   ||____________\
        Values

Left-skewed (tail on left):
Frequency
         ∧
        /||
       / ||
      /  ||
     /   ||
    /    ||
___/     ||
         Values
```
**Example:** Income (right-skewed), Age at retirement (left-skewed)

### **Why analyze distributions?**
- **Choose right statistical tests**: Some need normal data
- **Identify transformation needs**: Skewed data might need log transformation
- **Understand data behavior**: Helps in feature engineering

### **Tools we use:**
- **Histograms**: Show frequency of value ranges
- **Box Plots**: Show median, quartiles, and outliers
- **KDE plots**: Smooth curve showing distribution shape

---

## 🔗 Correlation Analysis

### **What is correlation?**
Measuring how much two variables change together - do they increase/decrease together?

### **Correlation Strength:**
```
+1.0: Perfect positive (as one goes up, other goes up)
+0.7: Strong positive
+0.3: Weak positive
 0.0: No relationship
-0.3: Weak negative
-0.7: Strong negative
-1.0: Perfect negative (as one goes up, other goes down)
```

### **Examples:**

#### **Positive Correlation (+0.8):**
```
Price vs Quality Rating:
Price:  [$10, $20, $30, $40, $50]
Rating: [2.1, 3.2, 4.1, 4.5, 4.8]
→ Higher price, higher rating
```

#### **Negative Correlation (-0.6):**
```
Price vs Number of Sales:
Price: [$10, $20, $30, $40, $50]
Sales: [100, 80,  60,  30,  10]
→ Higher price, fewer sales
```

#### **No Correlation (0.1):**
```
Product Color vs Price:
Color: [Red, Blue, Green, Yellow, Black]
Price: [$25,  $30,   $20,    $35,   $28]
→ No clear pattern
```

### **How to interpret correlation heatmap:**
```
        Price  Rating  Sales
Price    1.0    0.8   -0.6
Rating   0.8    1.0   -0.3
Sales   -0.6   -0.3    1.0
```
- **Diagonal = 1.0**: Variable correlated with itself
- **0.8**: Strong positive correlation between Price and Rating
- **-0.6**: Moderate negative correlation between Price and Sales

### **Important note:**
**Correlation ≠ Causation**
- High correlation doesn't mean one causes the other
- Could be coincidence or both influenced by third factor

---

## 📈 Statistical Hypothesis Testing

### **What is hypothesis testing?**
A formal way to test if what we observe in our data is real or just happened by chance.

### **Basic concept:**
1. **Make a claim** (hypothesis)
2. **Collect evidence** (data)
3. **Test the claim** (statistical test)
4. **Make a decision** (accept or reject)

### **Example: T-Test**

#### **Question:** "Do expensive products have higher ratings than cheap products?"

#### **Setup:**
```python
# Expensive products (>$50):
Ratings: [4.2, 4.5, 4.1, 4.7, 4.3]
Average: 4.36

# Cheap products (<$20):
Ratings: [3.1, 3.5, 2.9, 3.2, 3.8]
Average: 3.30
```

#### **Hypotheses:**
- **H0 (Null)**: No difference in ratings between expensive and cheap products
- **H1 (Alternative)**: Expensive products have higher ratings

#### **Test Result:**
```python
T-statistic: 5.23
P-value: 0.003

Interpretation:
- P-value < 0.05 → Statistically significant
- Conclusion: Reject H0, expensive products DO have higher ratings
```

### **Understanding P-values:**
- **P-value**: Probability that results happened by chance
- **P < 0.05**: Less than 5% chance it's random → Result is "significant"
- **P > 0.05**: More than 5% chance it's random → Result is "not significant"

---

## 📏 Distribution Fitting

### **What is distribution fitting?**
Finding which mathematical pattern best describes your data.

### **Common distributions:**

#### **1. Normal Distribution**
- **Shape**: Bell curve (symmetric)
- **When to use**: Heights, weights, test scores
- **Parameters**: Mean (center), Standard deviation (spread)

#### **2. Exponential Distribution**
- **Shape**: Steep decline from left
- **When to use**: Time between events, customer wait times
- **Example**: Time between website visits

#### **3. Poisson Distribution**
- **Shape**: Discrete (whole numbers only)
- **When to use**: Counting events (reviews per day, defects per batch)
- **Example**: Number of customer complaints per day

#### **4. Log-Normal Distribution**
- **Shape**: Right-skewed
- **When to use**: Prices, income, file sizes
- **Example**: Product prices (few expensive, many cheap)

### **How to choose best fit:**
1. **Visual inspection**: Plot data vs fitted curves
2. **AIC (Akaike Information Criterion)**: Lower = better fit
3. **Statistical tests**: Formal tests for goodness of fit

### **Example:**
```python
Price distribution analysis:
- Normal fit: AIC = 1250.5
- Log-normal fit: AIC = 1180.2  ← Better fit (lower AIC)
- Exponential fit: AIC = 1320.8

Conclusion: Product prices follow log-normal distribution
```

---

## 🎯 Outlier Detection Methods

### **What are outliers?**
Data points that are unusually different from the rest.

### **Methods to detect outliers:**

#### **1. IQR (Interquartile Range) Method**
```python
Step 1: Find quartiles
Q1 (25th percentile): $20
Q3 (75th percentile): $60
IQR = Q3 - Q1 = $40

Step 2: Calculate bounds
Lower bound = Q1 - 1.5×IQR = $20 - $60 = -$40
Upper bound = Q3 + 1.5×IQR = $60 + $60 = $120

Step 3: Find outliers
Prices: [$15, $25, $45, $55, $200]
                              ↑
                         Outlier! (>$120)
```

#### **2. Z-Score Method**
```python
Z-score = (Value - Mean) / Standard Deviation

Example:
Mean price: $50
Standard deviation: $20
Price of $130: Z-score = (130-50)/20 = 4.0

Rule: |Z-score| > 3 = Outlier
4.0 > 3 → Outlier!
```

#### **3. Modified Z-Score Method**
- **More robust**: Uses median instead of mean
- **Better for**: Data with many outliers
- **Threshold**: |Modified Z-score| > 3.5

### **Visual methods:**
- **Box Plot**: Shows outliers as dots beyond whiskers
- **Scatter Plot**: Outliers appear far from main cluster
- **Histogram**: Outliers appear as isolated bars

### **What to do with outliers:**
1. **Investigate**: Is it an error or legitimate value?
2. **Business context**: $200 shirt could be designer brand
3. **Actions**:
   - Remove if data error
   - Keep if legitimate extreme value
   - Transform data (log scale) to reduce impact

---

## 🧪 Statistical Tests Explained

### **1. T-Test**
**Purpose**: Compare means of two groups

**Example**: Do men and women rate products differently?
```python
Men's ratings: [4.1, 4.3, 4.0, 4.2]
Women's ratings: [4.5, 4.7, 4.4, 4.6]

Question: Is the difference significant?
```

**Types:**
- **One-sample t-test**: Compare sample mean to known value
- **Two-sample t-test**: Compare means of two groups
- **Paired t-test**: Compare before/after measurements

---

### **2. Chi-Square Test**
**Purpose**: Test if two categorical variables are related

**Example**: Is product category related to availability?
```python
Contingency Table:
           In Stock  Out of Stock
T-Shirts      80         20
Jeans         70         30
Dresses       60         40

Question: Are some categories more likely to be out of stock?
```

**Interpretation:**
- **H0**: Category and availability are independent
- **H1**: Category and availability are related
- **Result**: If p-value < 0.05, categories affect availability

---

### **3. Normality Tests**

#### **Shapiro-Wilk Test**
- **Purpose**: Test if data is normally distributed
- **Best for**: Small samples (<5000)
- **Interpretation**: p-value < 0.05 → data is NOT normal

#### **Kolmogorov-Smirnov Test**
- **Purpose**: Compare sample to theoretical distribution
- **Best for**: Larger samples
- **Use**: Test if data follows specific distribution

#### **D'Agostino Test**
- **Purpose**: Test for normality using skewness and kurtosis
- **Advantage**: Works well for various sample sizes

---

## 📊 Data Visualization Techniques

### **1. Count Plots**
**Purpose**: Show frequency of categories
```python
# Bar chart showing:
T-Shirts: ████████████ (250)
Jeans:    ██████████ (200)
Dresses:  ████████ (150)
Shoes:    ██████████████ (300)
```

### **2. Histograms**
**Purpose**: Show distribution of numerical data
```python
# Price distribution:
$0-20:   ████
$20-40:  ████████████
$40-60:  ████████
$60-80:  ████
$80-100: ██
```

### **3. Box Plots**
**Purpose**: Show median, quartiles, and outliers
```python
Components:
- Box: 25th to 75th percentile
- Line in box: Median
- Whiskers: 1.5×IQR from box edges
- Dots: Outliers
```

### **4. Heatmaps**
**Purpose**: Show correlation matrix as color-coded grid
```python
Correlation strength:
Dark red: Strong positive (+0.8 to +1.0)
Light red: Weak positive (+0.1 to +0.3)
White: No correlation (around 0)
Light blue: Weak negative (-0.1 to -0.3)
Dark blue: Strong negative (-0.8 to -1.0)
```

---

## 🛠️ Tools and Libraries

### **Matplotlib**
- **Purpose**: Basic plotting
- **Strengths**: Highly customizable, fine control
- **Use for**: Histograms, line plots, scatter plots

### **Seaborn**
- **Purpose**: Statistical visualization
- **Strengths**: Beautiful defaults, statistical functions
- **Use for**: Box plots, heatmaps, distribution plots

### **Plotly**
- **Purpose**: Interactive visualizations
- **Strengths**: Zoom, hover, interactive features
- **Use for**: Complex dashboards, web applications

### **SciPy**
- **Purpose**: Statistical functions and tests
- **Strengths**: Comprehensive statistical toolkit
- **Use for**: T-tests, normality tests, distribution fitting

### **Statsmodels**
- **Purpose**: Advanced statistical modeling
- **Strengths**: Detailed statistical output
- **Use for**: Regression analysis, time series

---

## 🎯 Key Statistical Concepts

### **1. Measures of Central Tendency**
- **Mean**: Average value (sum ÷ count)
- **Median**: Middle value when sorted
- **Mode**: Most frequently occurring value

**When to use which:**
- **Normal data**: Mean is best
- **Skewed data**: Median is more robust
- **Categorical data**: Mode is appropriate

### **2. Measures of Spread**
- **Standard Deviation**: Average distance from mean
- **Variance**: Standard deviation squared
- **Range**: Maximum - minimum value
- **IQR**: 75th percentile - 25th percentile

### **3. Skewness and Kurtosis**
- **Skewness**: Measure of asymmetry
  - Positive: Right tail longer
  - Negative: Left tail longer
  - Zero: Symmetric
- **Kurtosis**: Measure of tail heaviness
  - High: Heavy tails, sharp peak
  - Low: Light tails, flat peak

---

## 💡 Practical Tips for EDA

### **1. Start Simple**
- Begin with basic statistics and histograms
- Gradually move to more complex analyses
- Always visualize before testing

### **2. Business Context Matters**
- Statistical significance ≠ business significance
- Consider practical importance of findings
- Domain knowledge guides interpretation

### **3. Multiple Perspectives**
- Use different visualizations for same data
- Cross-validate findings with different methods
- Look for consistent patterns across analyses

### **4. Document Everything**
- Record assumptions and decisions
- Explain reasoning for choices
- Keep track of insights and questions

---

## 🔍 Common EDA Mistakes to Avoid

### **1. Data Snooping**
- **Problem**: Testing too many hypotheses without correction
- **Solution**: Plan analyses beforehand, adjust p-values for multiple testing

### **2. Correlation Confusion**
- **Problem**: Assuming correlation implies causation
- **Solution**: Use domain knowledge, consider alternative explanations

### **3. Outlier Obsession**
- **Problem**: Removing outliers without investigation
- **Solution**: Understand why outliers exist before removing

### **4. Ignoring Assumptions**
- **Problem**: Using tests without checking requirements
- **Solution**: Verify normality, independence, etc. before testing

---

## 🎯 Key Takeaways

### **EDA helps you:**
1. **Understand your data** before modeling
2. **Find patterns and relationships** that inform strategy
3. **Detect problems** early in the process
4. **Make informed decisions** about preprocessing and modeling
5. **Communicate insights** to stakeholders

### **Remember:**
- EDA is iterative - keep exploring and questioning
- Visualization is powerful - a good plot is worth 1000 statistics
- Context matters - statistical significance must make business sense
- Document your journey - insights build on each other

### **Best practices:**
- Start with questions, not just data
- Use multiple visualization types
- Test statistical assumptions
- Consider business implications
- Validate findings across different subsets

---

## 📚 Further Reading
- [Exploratory Data Analysis Guide](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)
- [Statistical Testing in Python](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Visualization Best Practices](https://www.storytellingwithdata.com/)
- [Understanding Correlation vs Causation](https://tylervigen.com/spurious-correlations)
