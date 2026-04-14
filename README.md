# 🚀 Student Depression Risk Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-FFD217?style=flat&logo=pandas&logoColor=150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-0077B6?style=flat&logo=python&logoColor=white)

## 📌 Overview

This project presents an innovative **early detection system for student depression risk** leveraging Machine Learning. Designed to empower educational institutions and counselors, the system analyzes a comprehensive set of student attributes—including behavioral, academic, and lifestyle patterns—to proactively identify individuals at risk. The core of the system is a robust Logistic Regression model, chosen for its interpretability, which is crucial for providing actionable insights in sensitive mental health contexts. The solution features a two-phase architecture: a meticulously crafted Model Development Pipeline (documented in a Jupyter Notebook) and a user-friendly Real-time Prediction Service built with Streamlit, enabling immediate risk assessment based on interactive user inputs.

## 🎯 Context & Problem Statement

The mental well-being of students is a critical concern, with depression being a significant impediment to academic success and overall quality of life. Traditional methods of identifying at-risk students are often reactive, relying on observable distress or self-reporting, which can lead to delayed intervention. This project addresses the urgent need for a **proactive, data-driven approach** to student mental health.

The primary business goal is to **minimize the incidence of undetected depression among students**, thereby facilitating timely support and preventing the escalation of mental health issues. By providing an interpretable risk assessment, the system enables counselors to understand the contributing factors for each student, allowing for more personalized and effective interventions. This not only improves student outcomes but also enhances the reputation and support infrastructure of educational institutions, demonstrating a commitment to holistic student welfare. The system translates directly into **reduced long-term health costs**, **improved academic performance**, and a **healthier, more engaged student body**.

## 📊 Quantitative Metrics

The developed model demonstrates strong performance in identifying students at risk of depression, with a particular emphasis on minimizing false negatives to ensure comprehensive coverage for intervention.

*   **Dataset Size**: The model was trained on a meticulously cleaned dataset of **27,876 records**, refined from an initial 28,008 raw entries.
*   **Features**: The final model utilizes **13 carefully selected features**, reduced from 16 original attributes through a rigorous feature importance analysis.
*   **Final Model Performance (Logistic Regression - Untuned)**:
    *   **Accuracy**: **85.01%** - Indicating a high overall correctness in predictions.
    *   **Recall**: **89%** - Crucially, this signifies that 89% of all actually depressed students are correctly identified by the model, minimizing the risk of missing at-risk individuals.
    *   **Precision**: **86%** - Meaning 86% of students predicted as depressed are indeed depressed.
    *   **F1-Score**: **87%** - A balanced measure of precision and recall.
*   **Top Predictive Features**: The model provides clear insights into the most influential factors:
    *   'Have you ever had suicidal thoughts?' (Absolute Coefficient: **2.472**)
    *   'Academic Pressure' (Absolute Coefficient: **1.151**)
    *   'Dietary Habits = Unhealthy' (Absolute Coefficient: **1.100**)

**Business Impact**: The exceptional **Recall of 89%** is the cornerstone of this system's value proposition. For an early detection system, **minimizing false negatives** (students who are depressed but classified as not depressed) is paramount. This high recall ensures that the vast majority of at-risk students are flagged, enabling proactive intervention and support. This translates directly into:
*   **Enhanced Student Well-being**: More students receive timely mental health support, preventing conditions from worsening.
*   **Improved Academic Outcomes**: Early intervention can mitigate the negative impact of depression on academic performance.
*   **Resource Optimization**: Counselors can focus their efforts on identified at-risk individuals, leading to more efficient resource allocation.
*   **Ethical Responsibility**: Demonstrates a strong commitment to student welfare and proactive health management.
The interpretability of Logistic Regression, backed by quantifiable feature coefficients, provides counselors with a clear understanding of *why* a student is flagged, facilitating targeted and empathetic interventions.

## 📷 Screenshots & Demo

This section provides a visual walkthrough of the application's user interface and key functionalities.

### 1. 🏡 Landing Interface
![Home UI](assets/home_ui.png)  
*The main landing page of the Streamlit application, featuring a dark-themed hero section with a gradient title and a clear description of the depression prediction service.*

### 2. 📝 Input Form
![Input Form](assets/input_form.png)  
*A detailed view of the input form where users provide 16 student attributes across demographic, academic, lifestyle, and psychological categories. Each field includes clear labels, default values, and helpful tooltips for ease of use.*

### 3. 🧠 Prediction Result
![Prediction Result](assets/prediction_result.png)  
*The prediction output card, dynamically styled based on the predicted status. It clearly displays the student's predicted depression status, the probability of depression, and the model's confidence score, offering nuanced insights.*

## ⚙️ Architecture & Data Flow

The system operates through a robust two-phase architecture: a **Model Development Pipeline** for continuous improvement and a **Real-time Prediction Service** for immediate user interaction.

### 1. 📊 Model Development Pipeline (Offline)

This phase, primarily executed within a Jupyter Notebook (`FAST_TRACK_Prediksi_Depresi_Bengkod_Final.ipynb`), focuses on the end-to-end process of building, evaluating, and persisting the machine learning model.

*   **Data Ingestion**: Raw student data (`Bengkod-Depresi.csv`) is loaded.
*   **Exploratory Data Analysis (EDA)**: Initial data quality checks, statistical summaries, and visualizations are performed to understand data distributions and identify potential issues.
*   **Data Cleaning & Preprocessing**: This iterative step involves:
    *   **Duplicate Handling**: Removing redundant records.
    *   **Missing Value Imputation**: Strategically filling missing numerical values with medians and categorical values with modes. Critical target variable missing values are dropped.
    *   **Anomalous Data Handling**: Correcting or removing illogical entries (e.g., `CGPA=0`, invalid `City` names, "///" in `Dietary Habits`).
    *   **Feature Engineering**: Grouping granular categorical features (`City`, `Degree`) into broader, more meaningful categories.
    *   **Feature Removal**: Dropping non-predictive features (`id`) and low-contribution features (`Gender`, `Job Satisfaction`, `Work Pressure`).
*   **Preprocessing Pipeline Definition**: A `ColumnTransformer` is configured to apply `OneHotEncoder` to categorical features and `StandardScaler` to numerical features, ensuring proper data transformation and preventing data leakage.
*   **Model Training & Evaluation**:
    *   Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, KNN, Naive Bayes) are trained within an `ImbPipeline` (integrating preprocessing steps).
    *   Models are rigorously evaluated using Accuracy, Recall, Precision, F1-score, and Confusion Matrices.
    *   Logistic Regression is selected for its balance of performance and interpretability.
*   **Hyperparameter Tuning**: `GridSearchCV` with `StratifiedKFold` is employed to optimize Logistic Regression, specifically targeting `recall`.
*   **Model Persistence**: The entire trained `ImbPipeline` object, including the `ColumnTransformer` and the final Logistic Regression model, is serialized using `joblib` and saved as `best_model.joblib`. This ensures that the exact preprocessing steps and model weights are preserved for deployment.

### 2. 🌐 Real-time Prediction Service (Online)

This phase provides an interactive web application (`app.py` and `function.py`) for users to obtain instant depression risk predictions.

*   **User Interface (Streamlit - `app.py`)**: A Streamlit application renders a modern, dark-themed web interface. It presents a form with 16 input fields for student attributes, enhanced with tooltips and default values.
*   **Input Data Transformation (`function.py`)**: User inputs from the Streamlit form are collected and transformed into a Pandas DataFrame. Crucially, the three insignificant columns (`Gender`, `Job Satisfaction`, `Work Pressure`) are explicitly dropped to match the feature set of the trained model.
*   **Model Loading (`function.py`)**: The `best_model.joblib` pipeline is loaded into memory using `joblib`. Streamlit's `@st.cache_resource` decorator optimizes this process, caching the model to prevent redundant loading and improve performance. Robust error handling ensures graceful failure if the model file is inaccessible.
*   **Inference (`function.py`)**: The prepared DataFrame is fed into the loaded model. The model performs both binary prediction (Depressed/Not Depressed) and probability prediction. The numerical outputs are mapped to human-readable labels and formatted for display.
*   **Result Presentation (`app.py`)**: The Streamlit app dynamically displays a custom-styled HTML card, presenting the predicted status, the probability of depression, and the model's confidence score to the user.

```mermaid
graph TD
    subgraph Model Development Pipeline (Offline)
        A["Raw Data (Bengkod-Depresi.csv)"] --> B("Data Ingestion & Initial EDA")
        B --> C{"Data Cleaning & Preprocessing"}
        C -- "Duplicates, Missing Values, Anomalies, Feature Engineering" --> D("Feature Selection")
        D --> E("Preprocessing Pipeline Definition (ColumnTransformer)")
        E --> F("Model Training (ImbPipeline: LR, RF, XGB, KNN, NB)")
        F --> G("Hyperparameter Tuning (GridSearchCV, recall)")
        G --> H("Model Evaluation (Accuracy, Recall, F1, CM)")
        H --> I("Final Model Selection (Logistic Regression)")
        I --> J[("best_model.joblib - Saved Pipeline")]
    end

    subgraph Real-time Prediction Service (Online)
        K["User Input (Streamlit UI)"] --> L("Input Data Transformation (DataFrame creation)")
        L --> M{"Drop Insignificant Features"}
        M --> N("Load Model (best_model.joblib) @st.cache_resource")
        N --> O("Perform Inference (predict_proba)")
        O --> P("Map Predictions & Format Results")
        P --> Q["Display Results (Streamlit Card)"]
    end

    J -- "Loaded into memory" --> N
```
*Note: This architecture diagram is AI-generated using Mermaid.js. If you encounter rendering issues on certain platforms, minor manual syntax adjustments (e.g., escaping special characters or fixing subgraph IDs) may be required.*

## 💻 Installation & Reproduction Steps

To set up the project and reproduce the results, follow these steps.

### 1. 📦 Prerequisites

Ensure you have the following installed:
*   Python 3.8+
*   `pip` (Python package installer)

### 2. ⬇️ Clone the Repository

Open your terminal or command prompt and execute the following commands to clone the project repository:

```bash
git clone https://github.com/viochris/Depression-Prediction.git
cd Depression-Prediction
```

### 3. 🐍 Install Dependencies

Navigate into the cloned directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. ⚙️ Model Training (Optional, for reproduction)

To reproduce the model training and evaluation, you will need to run the Jupyter Notebook.

*   **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
*   **Open Notebook:** In your browser, navigate to and open `FAST_TRACK_Prediksi_Depresi_Bengkod_Final.ipynb`.
*   **Run All Cells:** Execute all cells in the notebook sequentially. This will perform data loading, preprocessing, model training, evaluation, and save the `best_model.joblib` file in the `Depression-Prediction-Model` directory.

### 5. ▶️ Run the Prediction Service

Once the `best_model.joblib` is available (either by running the notebook or if it's already in the repository), you can launch the Streamlit application:

```bash
streamlit run app.py
```

This command will open the Streamlit application in your default web browser, typically at `http://localhost:8501`. You can then interact with the web interface to input student data and receive real-time depression risk predictions.

## ⚠️ System Limitations & Future Work

While the Student Depression Risk Predictor offers a robust and valuable solution, it's important to acknowledge its current limitations and potential areas for future enhancement.

### 1. 🏛️ Architectural Limitations (Inherent in Design)

1.  **Single-Server Streamlit Application**:
    *   **Scalability**: The current Streamlit setup is designed for single-instance deployment. It lacks inherent horizontal scalability, meaning it would struggle to handle a large volume of concurrent users without significant architectural changes (e.g., containerization, load balancing, or a more robust web framework like FastAPI/Flask).
    *   **Availability**: It represents a single point of failure. If the server hosting the Streamlit app goes down, the prediction service becomes unavailable.
    *   **Performance**: All computations occur on the same server, which can lead to performance bottlenecks under heavy load.
2.  **In-Memory Model Loading**: The `best_model.joblib` is loaded into memory and cached. While efficient for repeated predictions, this implies:
    *   **Memory Footprint**: The model consumes server memory, which could become a concern for larger, more complex models or when running multiple services on the same machine.
    *   **Cold Start Latency**: The very first request after deployment or cache invalidation will experience higher latency due to the initial model loading time.
3.  **Synchronous Processing**: Streamlit applications are predominantly synchronous. A long-running prediction task for one user could potentially block other users' requests, degrading the overall user experience.
4.  **Hardcoded File Paths**: The model loading path (`"Depression-Prediction-Model/best_model.joblib"`) is hardcoded within `function.py`. This reduces deployment flexibility, requiring a specific directory structure and making containerization or cloud deployment less straightforward without code modification or environment variable integration.
5.  **Lack of API Endpoint**: The Streamlit application provides a GUI but does not expose a RESTful API. This limits its integration capabilities, requiring a separate API layer to be built if other systems or applications need programmatic access to the prediction service.

### 2. ⏳ Runtime Limitations (Specific to Data & Development Workflow)

1.  **"Fast-Track" Development Imperfections**:
    *   **Suboptimal Workflow**: The notebook's "Direct Modelling" phase, conducted *before* comprehensive data cleaning and preprocessing, means baseline models were trained on noisy data. This makes the comparison between baseline and final models less "apple-to-apple" and potentially overstates the impact of preprocessing.
    *   **Misleading EDA**: Initial EDA was performed on raw, uncleaned data. Consequently, some early visualizations (e.g., for `City`, `Degree`, `Dietary Habits`) contained misleading insights due to unstandardized or anomalous values (e.g., people's names in `City`).
2.  **Model Selection Rationale**: Logistic Regression was primarily chosen for its interpretability, which is valuable for counseling contexts. However, other models like Random Forest and XGBoost showed comparable (or slightly better in some metrics, prior to tuning) performance. This indicates a trade-off between maximizing raw predictive power and maintaining model interpretability.
3.  **Geographical Generalizability**: The dataset is explicitly India-centric, featuring Indian city names and social contexts. The model's ability to generalize to students in other countries (e.g., Indonesia, as per some context mentions) is unvalidated and would likely require retraining or significant re-validation with local data.
4.  **Heavy Reliance on "Suicidal Thoughts" Feature**: The model exhibits a strong dependency on the 'Have you ever had suicidal thoughts?' feature, which has a significantly higher coefficient (2.472) than all others.
    *   **Ethical Concerns**: Requiring this as a mandatory input in a real-world application raises substantial ethical concerns regarding privacy, data sensitivity, and the potential for misinterpretation or misuse without proper clinical guidance and context.
    *   **Data Availability**: In practical scenarios, obtaining accurate and consistent data on suicidal ideation can be challenging or ethically problematic.
5.  **No Versioning for Model/Data**: The project lacks explicit mechanisms for model versioning or data versioning (e.g., DVC, MLflow). This makes it difficult to track changes in the model or data over time, reproduce past results, or manage different model deployments effectively in a production environment.

---
**Author:** [Silvio Christian, Joe](https://github.com/viochris)
*Empowering proactive mental health support through intelligent data insights.*
