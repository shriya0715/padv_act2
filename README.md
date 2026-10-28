# Software Defect Prediction Using Machine Learning and Interactive Analytics Dashboard

An end-to-end academic machine learning application and interactive analytics dashboard designed to predict software defects using historical static code metrics from the NASA Metrics Data Program (MDP).

Built with **Python**, **Streamlit**, **Scikit-Learn**, **Pandas**, and **Plotly**.

---

## 📌 Project Overview & Problem Statement

Software testing and quality assurance (QA) consume up to **50–80%** of total software development lifecycle costs. In large codebases, exhaustive testing of every subroutine or module is virtually impossible due to time and resource constraints. 

This project implements machine learning models—specifically **Decision Tree** and **Support Vector Machine (SVM)** classifiers—trained on static software engineering metrics (McCabe Cyclomatic Complexity and Halstead Software Science measures) to predict whether software modules are **defective** (`True`) or **defect-free** (`False`). 

By accurately identifying high-risk software modules before deployment, QA teams can prioritize code reviews, focus automated unit and mutation tests, and mitigate severe bugs escaping to production.

---

## 🎯 Key Objectives

1. **Robust Data Hygiene**: Ingest and clean NASA MDP software defect datasets (`cm1.csv` and `jm1.csv`), converting malformed strings, handling missing values, and detecting identical metric profiles without silent data loss.
2. **Leak-Free Preprocessing**: Implement strict scikit-learn pipelines where imputation and feature scaling are fitted strictly on training data folds.
3. **Exploratory Data Analysis (EDA)**: Provide interactive visualizations for defect distributions, skewed metric histograms, box plots, Pearson correlation matrices, and empirical risk curves.
4. **Machine Learning Classification**: Train, evaluate, and compare **Decision Tree** and **Support Vector Machine (SVM)** classifiers.
5. **Rigorous Evaluation**: Evaluate models with a primary focus on **Recall**, **Precision**, and **F1-Score** to address severe real-world class imbalance, alongside ROC-AUC and Precision-Recall (PR) curves.
6. **Feature Importance & Interpretability**: Extract Gini importance from Decision Trees and compute Permutation Importance on held-out test data for SVM models.
7. **Interactive Single-Module Inference**: Provide a live diagnostic tool allowing engineers to input code metrics and receive immediate defect likelihood percentages and risk tiers.
8. **Actionable QA Recommendations & Academic Defense**: Deliver evidence-based engineering actions and document critical academic project limitations.

---

## 📊 Dataset Description

The project operates on two benchmark datasets from the **NASA Metrics Data Program (MDP)**:

| Dataset | Total Modules | Defective Modules | Defect Rate | Domain / Subsystem | Language |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **CM1** | 498 | 49 | 9.84% | Spacecraft instrument subsystem | C / C++ |
| **JM1** | 10,885 | 2,106 | 19.35% | Real-time ground tracking system | C / C++ |

### Software Metrics Dictionary

Both datasets contain 21 numerical predictor metrics and 1 binary target:

#### 1. McCabe Sizing and Complexity Metrics
* `loc`: Lines of Code (total module size)
* `v(g)`: Cyclomatic Complexity (number of linearly independent control paths; standard threshold $\le 10$)
* `ev(g)`: Essential Complexity (degree of unstructured control flow / spaghetti code)
* `iv(g)`: Design Complexity (module coupling and call-graph complexity)
* `branchCount`: Number of decision branch exits in the control flow graph

#### 2. Line Composition Metrics
* `lOCode`: Executable lines of code
* `lOComment`: Lines of comments
* `lOBlank`: Blank whitespace lines
* `locCodeAndComment`: Lines containing both code and comments

#### 3. Halstead Software Science Metrics
* `uniq_Op` ($\mu_1$): Number of distinct operators
* `uniq_Opnd` ($\mu_2$): Number of distinct operands (variables and constants)
* `total_Op` ($N_1$): Total occurrences of operators
* `total_Opnd` ($N_2$): Total occurrences of operands
* `n` ($N$): Total program vocabulary length ($N_1 + N_2$)
* `v` ($V$): Program Volume in bits ($V = N \times \log_2(\mu)$)
* `l` ($L$): Program Level / Abstraction ($L = V^* / V$)
* `d` ($D$): Difficulty to write and understand ($D = \frac{\mu_1}{2} \times \frac{N_2}{\mu_2}$)
* `i` ($I$): Intelligence content independent of language ($I = V / D$)
* `e` ($E$): Mental Effort to comprehend the module ($E = V \times D$)
* `b` ($B$): Halstead estimated errors / bugs ($B = V / 3000$)
* `t` ($T$): Implementation time estimate in seconds ($T = E / 18$)

#### 4. Ground Truth Target
* `defects`: Binary target (`True` = defective, `False` = defect-free)
* `id`: Unique module identifier (preserved for inspection; strictly excluded from model feature sets)

---

## 🛠️ Technologies & Libraries

* **Python 3.10+**: Core programming language.
* **Streamlit**: Web application framework and dashboard interface.
* **Pandas**: Tabular data manipulation, aggregation, and filtering.
* **NumPy**: Vectorized numerical operations and linear algebra.
* **Scikit-Learn**: Machine learning pipelines, classifiers (`DecisionTreeClassifier`, `SVC`), preprocessing (`SimpleImputer`, `StandardScaler`), and evaluation metrics.
* **Plotly Express & Graph Objects**: Interactive charts with zoom, hover tooltips, and responsive layout.
* **Joblib**: Model serialization and disk persistence.
* **Pathlib**: Robust, cross-platform file path resolution.

---

## 📁 Project Directory Structure

```text
padb/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and project guide
├── .gitignore                 # Version control exclusions
├── test_core.py               # Automated unit & integration test suite
├── test_app_startup.py        # Streamlit server startup & HTTP 200 probe test
│
├── data/                      # NASA MDP datasets
│   ├── cm1.csv                # CM1 dataset (498 records)
│   └── jm1.csv                # JM1 dataset (10,885 records)
│
├── src/                       # Modular source code
│   ├── __init__.py            # Package initializer
│   ├── data_loader.py         # Dataset loading, validation, and metric metadata
│   ├── preprocessing.py       # Data sanitization, string conversion, and leak-free splitting
│   ├── eda.py                 # Interactive Plotly visualization generators
│   ├── model_training.py      # Decision Tree & SVM pipelines with persistence
│   ├── evaluation.py          # Metric calculations, confusion matrices, ROC & PR curves
│   ├── feature_analysis.py    # Gini feature importance & SVM permutation importance
│   ├── prediction.py          # Single-module inference, risk tiering, and validation
│   └── recommendations.py     # Dynamic QA recommendations and academic limitations
│
└── models/                    # Saved model artifacts (.joblib) and metadata (.json)
    ├── cm1_decision_tree.joblib
    ├── cm1_decision_tree_meta.json
    ├── cm1_support_vector_machine.joblib
    └── ...
```

---

## 🚀 Installation & Setup in Visual Studio Code (VS Code)

Follow these exact steps in **VS Code** on Windows:

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File** ➔ **Open Folder...**
3. Select the `padb` folder (located at `C:\Users\Shreeya\OneDrive\Desktop\padb`).
4. Open the integrated terminal by pressing ``Ctrl + ` `` (or **Terminal** ➔ **New Terminal**). Ensure PowerShell is active.

### Step 2: Create a Virtual Environment
In the PowerShell terminal, run:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Verify the Installation & Run Tests
Execute the automated test suite to confirm data loading, preprocessing, model pipelines, and predictions:
```powershell
python test_core.py
```
You should see:
```text
============================================================
ALL CORE UNIT AND INTEGRATION TESTS COMPLETED SUCCESSFULLY!
============================================================
```

### Step 6: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch automatically in your default web browser at:
👉 **`http://localhost:8501`**

---

## 🖥️ Using the Interactive Dashboard

The dashboard is structured into **7 navigation sections** accessible via the sidebar:

### 1. 📊 Overview
* **KPI Metric Cards**: Real-time summary of Total Modules, Defective Modules, Defect-Free Modules, Metrics Tracked, and Best Model F1-Score.
* **Class Distribution**: Interactive bar and donut charts illustrating class imbalance.
* **Dataset Scale Comparison**: Side-by-side comparison of CM1 and JM1 dataset volumes and defect percentages.
* **Data Hygiene Summary**: Transparency card detailing original vs. cleaned dimensions, converted string columns, and duplicate profiles.

### 2. 🔍 Dataset Explorer
* **Searchable Data Table**: View, sort, and select individual feature columns.
* **Interactive Filters**: Filter modules dynamically by defect status (`Defective Only`, `Defect-Free Only`) or maximum lines of code (`loc`).
* **Descriptive Statistics**: One-click summary statistics (mean, std, min, 25%, median, 75%, max) across all software metrics.
* **Metric Dictionary**: Full mathematical formulations and domain families for McCabe and Halstead measures.
* **CSV Export**: Download filtered subsets as CSV without modifying the underlying source files.

### 3. 📈 Exploratory Data Analysis (EDA)
* **Univariate Distributions & Box Plots**: Interactive histograms with overlaid KDE and box plots. Includes a log-scale toggle for highly skewed metrics (e.g., Halstead Effort or Volume).
* **Bivariate Scatter Plots**: Compare any two metrics (e.g., `loc` vs. `v(g)`) with defect-status color coding.
* **Correlation Heatmap**: Pearson correlation matrix with optional inclusion of the target variable.
* **Empirical Defect Rate Curves**: Quantile risk analysis showing how defect rates rise across increasing metric ranges, complete with sample sizes per bin.
* **Defective vs. Clean Comparison**: Tabular ratios of mean and median metrics for defective vs. clean routines.

### 4. 🧠 Defect Prediction (Single-Module Testing)
* **Model Selection**: Switch between the trained **Decision Tree** and **Support Vector Machine**.
* **Metric Presets**: Instantly load median presets, high-complexity monolithic profiles, or clean stub profiles.
* **Logical Input Grouping**: Enter values across Sizing, McCabe Complexity, Halstead Token Counts, and Halstead Derived Measures.
* **Real-Time Inference**:
  * Visual status indicator: `⚠️ DEFECTIVE MODULE DETECTED` or `✅ DEFECT-FREE MODULE`.
  * Defect likelihood gauge and percentage score.
  * Risk tiering: **Low Risk** (<35%), **Moderate Risk** (35%–60%), or **High Risk** (>60%).
  * Specific diagnostic warnings (e.g., McCabe cyclomatic complexity $v(g) > 10$, lines of code $> 100$).

### 5. 🎯 Model Performance
* **Comparison Matrix**: Side-by-side table displaying Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision, True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).
* **Grouped Metric Bar Chart**: Direct visual benchmark of Decision Tree vs. SVM.
* **Interactive Confusion Matrices**: Labeled quadrants highlighting Caught Defects (TP), Missed Defects (FN), False Alarms (FP), and Correctly Clean (TN).
* **ROC & Precision-Recall Curves**: Threshold curves with chance baselines and empirical defect rate references.
* **Classification Reports**: Raw precision, recall, and f1-score per class.

### 6. 🔬 Feature Analysis
* **Decision Tree Gini Importance**: Normalized Gini impurity decrease sorted in descending order.
* **SVM Permutation Importance**: Evaluated on held-out test data by measuring the drop in F1-score when feature values are permuted.
* **Adjustable Top-N Slider**: Display top 5 to 21 features with interactive tooltips explaining metric families.
* **Domain Explanation**: Detailed briefing on why McCabe and Halstead measures carry high predictive signal.

### 7. 💡 Insights & Recommendations
* **Dynamic QA Recommendations**: Prioritized engineering action cards (High/Medium/Low priority) conditioned on actual dataset distributions (e.g., cyclomatic complexity gates, monolithic refactoring, risk-based test triage).
* **Academic Defense Briefing**: Bullet points addressing critical analytical nuances (dataset age, class imbalance, correlation vs. causation, cross-project variance, and false negative safety risks).

---

## 📈 Interpreting Evaluation Metrics (Student Guide)

When presenting this project to your professor or evaluator, use this guide:

| Metric | Formula | Why It Matters in Software Defect Prediction |
| :--- | :---: | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **Can be misleading under class imbalance.** On CM1 (~9.8% defects), a naive model predicting every module is clean gets 90.2% accuracy while catching zero bugs. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **Most critical metric for QA.** Measures what proportion of actual bugs the model caught. A low recall means bugs escape into production (False Negatives). |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures what proportion of predicted bugs were real. High precision minimizes "false alarms" that waste engineers' time during code review. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall. Best single metric for balancing defect detection against false alarms on imbalanced data. |
| **ROC-AUC** | Area under ROC Curve | Measures class discrimination across all possible classification probability thresholds ($0.5 = \text{random chance}$, $1.0 = \text{perfect}$). |
| **Average Precision** | Area under PR Curve | Summarizes the Precision-Recall tradeoff; especially informative when positive defects are rare. |

---

## ⚠️ Academic Project Limitations & Defense Briefing

Include these points in your project report and viva defense:

1. **Historical Context of NASA MDP**: CM1 and JM1 were compiled in the late 1990s and early 2000s from aerospace C/C++ codebases. While McCabe and Halstead metrics remain timeless software engineering foundations, modern development practices (e.g., garbage-collected languages, microservices, containerization, static type checkers) introduce different defect profiles.
2. **Correlation vs. Causation**: High lines of code (`loc`) and high cyclomatic complexity (`v(g)`) exhibit strong statistical correlation with defect occurrence, but they are **proxies**, not direct causes. Root causes often stem from ambiguous requirements, changing specifications, and human developer fatigue.
3. **Class Imbalance Realities**: Because defects represent a minority class (~9.8% in CM1, ~19.4% in JM1), classifiers must utilize class weighting (`class_weight='balanced'`) to prevent majority-class bias.
4. **The High Cost of False Negatives**: In mission-critical or safety-critical software, a False Negative (missed bug) is far more hazardous than a False Positive (a harmless code review alert).

---

## 🔧 Troubleshooting Common Issues

* **Error: `Missing required target column 'defects'`**:
  Ensure `cm1.csv` and `jm1.csv` are located inside the `data/` folder and have not been renamed.
* **Error: `ExecutionPolicy` restricted in PowerShell**:
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell terminal, then re-activate the virtual environment.
* **Port 8501 is already in use**:
  Run Streamlit on an alternative port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```
* **Models retraining slowly on JM1**:
  JM1 has 10,885 records. The Decision Tree fits in milliseconds; SVM with RBF kernel and probability calibration takes ~4–6 seconds. Streamlit caches the trained models in session state, so subsequent interactions are instantaneous.

---

## 📜 License & Academic Integrity

This project is developed for educational and academic demonstration purposes using open-source research data from the NASA Metrics Data Program (MDP). You are free to reference, adapt, and build upon this codebase for academic coursework and thesis presentations.
