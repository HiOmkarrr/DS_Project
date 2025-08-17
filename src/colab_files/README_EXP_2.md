# 📊 Experiment 2: Data Preprocessing & Validation - Concepts Explained

## 🎯 Overview
This README explains all the data science concepts used in Experiment 2 (DS_EXP_2.ipynb) in simple terms with practical examples.

---

## 📋 Table of Contents
1. [Data Profiling](#-data-profiling)
2. [Data Cleaning](#-data-cleaning)
3. [Feature Engineering](#-feature-engineering)
4. [Schema Validation](#-schema-validation)
5. [Data Version Control (DVC)](#-data-version-control-dvc)
6. [Missing Value Handling](#-missing-value-handling)
7. [Outlier Detection](#-outlier-detection)

---

## 📈 Data Profiling

### **What is it?**
Data profiling is like getting a "health checkup" for your dataset. It tells you the basic information about your data.

### **Why is it important?**
- Understand what you're working with
- Spot potential problems early
- Make informed decisions about cleaning

### **What we check:**
```python
# Basic information we gather:
- Dataset shape (rows × columns)
- Data types (numbers, text, dates)
- Missing values count
- Duplicate records
- Statistical summary
```

### **Example:**
```python
# Input: Clothing dataset
Dataset Shape: (1240, 15)  # 1240 products, 15 features
Missing Values: 23 in 'size' column
Duplicate Rows: 5 duplicates found
```

### **Real-world analogy:**
Like checking a library - counting books, checking for damaged pages, noting missing books.

---

## 🧹 Data Cleaning

### **What is it?**
Data cleaning is fixing problems in your dataset to make it ready for analysis.

### **Common problems we fix:**

#### **1. Missing Values**
- **Problem:** Some data points are empty
- **Solution:** Fill with appropriate values
```python
# Example:
Before: [Size: M, L, ?, XL, ?]
After:  [Size: M, L, M, XL, M]  # Filled with most common size
```

#### **2. Duplicate Records**
- **Problem:** Same product listed multiple times
- **Solution:** Keep only unique records
```python
# Example:
Before: 1245 products (5 duplicates)
After:  1240 unique products
```

#### **3. Inconsistent Formatting**
- **Problem:** Same data written differently
- **Solution:** Standardize format
```python
# Example:
Before: ['red', 'Red', ' RED ', 'red ']
After:  ['Red', 'Red', 'Red', 'Red']
```

### **Tools we use:**
- **Pandas**: For basic cleaning operations
- **Janitor**: For advanced cleaning functions

---

## ⚙️ Feature Engineering

### **What is it?**
Creating new, useful information from existing data to help your models work better.

### **Types of features we create:**

#### **1. Mathematical Features**
```python
# Creating discount value:
discount_value = original_price - current_price

# Example:
Original Price: $100
Current Price: $80
Discount Value: $20
```

#### **2. Binary Flags (Yes/No features)**
```python
# Creating discount flag:
discount_flag = 1 if discount_value > 0 else 0

# Example:
Discount Value: $20 → Discount Flag: 1 (Yes, on sale)
Discount Value: $0  → Discount Flag: 0 (No discount)
```

#### **3. Time-based Features**
```python
# Extracting day of week from date:
scraped_date = '2025-08-17'
scraped_weekday = 'Saturday'

# Use: Different buying patterns on weekends vs weekdays
```

#### **4. Categorical Binning**
```python
# Grouping prices into categories:
Price $300  → 'Morning' (Low)
Price $800  → 'Afternoon' (Medium)  
Price $1500 → 'Evening' (High)
```

### **Why do we do this?**
- Help models understand patterns better
- Create meaningful categories
- Capture business logic in data

---

## ✅ Schema Validation

### **What is it?**
Checking if your data follows the rules you expect, like a quality control inspector.

### **What we validate:**

#### **1. Data Types**
```python
# Rule: Price should be a number
✅ Valid: price = 299.99
❌ Invalid: price = "expensive"
```

#### **2. Value Ranges**
```python
# Rule: Rating should be between 0 and 5
✅ Valid: rating = 4.2
❌ Invalid: rating = 7.5
```

#### **3. Required Fields**
```python
# Rule: Category cannot be empty
✅ Valid: category = "T-Shirt"
❌ Invalid: category = null
```

### **Example Schema:**
```python
Schema Rules:
- price: Must be positive number
- category: Must be text, cannot be empty
- rating: Must be between 0-5
- discount_percentage: Must be between 0-100
```

### **Benefits:**
- Catch errors early
- Ensure data quality
- Prevent model failures

---

## 📦 Data Version Control (DVC)

### **What is it?**
Like "Save As" for datasets - keeping track of different versions of your data as it changes.

### **Why is it needed?**
- **Datasets are large**: Can't store in regular Git
- **Track changes**: Know what changed between versions
- **Collaboration**: Share datasets with team members
- **Reproducibility**: Go back to previous versions

### **How it works:**

#### **Step 1: Track your dataset**
```bash
dvc add datasets/cleaned_clothing_dataset.csv
# Creates: cleaned_clothing_dataset.csv.dvc (small pointer file)
```

#### **Step 2: Store the pointer in Git**
```bash
git add cleaned_clothing_dataset.csv.dvc
git commit -m "Add cleaned dataset v1.0"
```

#### **Step 3: Store actual data elsewhere**
```bash
dvc push  # Uploads to cloud storage (Google Drive, S3, etc.)
```

### **Real-world analogy:**
- **Git**: Like a library catalog (tracks book information)
- **DVC**: Like a warehouse (stores the actual books)
- **DVC file**: Like a receipt telling you where your book is stored

### **Example workflow:**
```
Version 1: Raw dataset (1000 products)
Version 2: Cleaned dataset (950 products, duplicates removed)
Version 3: Enhanced dataset (950 products + 50 new features)
```

---

## 🔍 Missing Value Handling

### **What are missing values?**
Empty spots in your data where information should be.

### **Why do they occur?**
- User didn't fill a form field
- Sensor malfunction
- Data transfer errors
- Optional information

### **Strategies to handle them:**

#### **1. Fill with Mode (Most Common Value)**
```python
# For categorical data like Size:
Missing sizes: [?, ?, ?]
Most common size: 'M'
After filling: ['M', 'M', 'M']
```

#### **2. Fill with Mean/Median**
```python
# For numerical data like Price:
Prices: [100, 200, ?, 150, ?]
Mean price: 150
After filling: [100, 200, 150, 150, 150]
```

#### **3. Remove Records**
```python
# If too many values are missing:
Before: 1000 products (100 with >50% missing data)
After: 900 products (removed problematic ones)
```

### **Decision factors:**
- **Amount missing**: <5% → Fill, >20% → Remove
- **Importance**: Critical field → Fill carefully
- **Data type**: Categories → Mode, Numbers → Mean/Median

---

## 📊 Outlier Detection

### **What are outliers?**
Data points that are very different from the rest - the "weird" values.

### **Example:**
```
T-shirt prices: [$15, $20, $18, $22, $25, $500]
                                              ↑
                                         Outlier!
```

### **Methods to find outliers:**

#### **1. Box Plot Method**
```python
# Visual method using quartiles:
Q1 (25th percentile): $18
Q3 (75th percentile): $25
IQR (Q3-Q1): $7
Lower bound: $18 - 1.5×$7 = $7.50
Upper bound: $25 + 1.5×$7 = $35.50

# $500 is way above $35.50 → Outlier!
```

#### **2. Histogram Analysis**
```python
# Visual method:
Most prices: Between $15-$30 (normal distribution)
One price: $500 (stands alone)
```

### **Why detect outliers?**
- **Data entry errors**: Someone typed 500 instead of 50
- **Special cases**: Luxury items vs regular items
- **Model impact**: Can skew predictions

### **What to do with outliers?**
1. **Investigate**: Is it an error or real?
2. **Remove**: If it's clearly wrong
3. **Transform**: Use log scale to reduce impact
4. **Keep**: If it represents real variation

---

## 🛠️ Tools and Libraries Used

### **Pandas**
- **Purpose**: Data manipulation and analysis
- **What it does**: Read files, clean data, transform data
- **Example**: `df.fillna()` fills missing values

### **Matplotlib & Seaborn**
- **Purpose**: Creating visualizations
- **What it does**: Make plots and charts
- **Example**: `sns.heatmap()` shows missing value patterns

### **Janitor**
- **Purpose**: Advanced data cleaning
- **What it does**: Clean column names, remove duplicates
- **Example**: `clean_names()` standardizes column names

### **Pandera**
- **Purpose**: Data validation
- **What it does**: Check if data follows rules
- **Example**: Validate that prices are positive numbers

---

## 🎯 Key Takeaways

### **Data Preprocessing is crucial because:**
1. **Garbage In = Garbage Out**: Clean data leads to better models
2. **Early detection**: Fix problems before they cause issues
3. **Consistency**: Standardized data is easier to work with
4. **Validation**: Ensure data meets business rules

### **Best practices:**
- Always profile your data first
- Document your cleaning steps
- Validate data quality regularly
- Version control your datasets
- Handle missing values thoughtfully
- Investigate outliers before removing them

### **Remember:**
Good data preprocessing can make the difference between a successful and failed data science project!

---

## 📚 Further Reading
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Data Cleaning Best Practices](https://towardsdatascience.com/data-cleaning-best-practices)
- [Feature Engineering Guide](https://towardsdatascience.com/feature-engineering-for-machine-learning)
