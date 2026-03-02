# Online Course Recommendation System

## Project Overview

This project develops an online course recommendation system using a combination of Collaborative Filtering (SVD, Item-Based CF) and Content-Based Filtering techniques. The goal is to provide personalized course suggestions to users based on their historical ratings and course content, enhancing the learning experience.

## Features

*   **Data Loading & Inspection**: Efficient loading of course data from an Excel file and initial examination.
*   **Data Cleaning**: Handling of missing values and duplicate entries to ensure data quality.
*   **Outlier Analysis**: Identification and capping of outliers in numerical features.
*   **Exploratory Data Analysis (EDA)**: Comprehensive visualization and analysis of data distributions, correlations, and categorical breakdowns.
*   **Feature Engineering & Encoding**: Transformation of categorical features into numerical representations suitable for machine learning models (Binary and Ordinal Encoding).
*   **Collaborative Filtering (SVD)**: Implementation of Singular Value Decomposition for user-item interaction based recommendations.
*   **Item-Based Collaborative Filtering (KNNWithMeans)**: Recommendations based on similarities between items.
*   **Content-Based Filtering**: Recommendations driven by course attributes (e.g., `course_name`, `difficulty_level`) using TF-IDF and cosine similarity.
*   **Hybrid Recommendation System**: A combined approach that leverages the strengths of all individual models for more robust and accurate recommendations.
*   **Model Evaluation**: Assessment of model performance using standard metrics like RMSE and MAE.
*   **Model Persistence**: Saving trained models for future use without re-training.

## Project Structure

The project workflow is structured as follows:

1.  **Data Loading & Inspection**: Initial steps to get the data into the notebook.
2.  **Data Cleaning**: Preprocessing steps to ensure data quality.
3.  **Outlier Analysis**: Handling extreme values in numerical data.
4.  **Exploratory Data Analysis**: Visualizing and understanding the dataset.
5.  **Feature Engineering & Encoding**: Preparing features for model training.
6.  **Recommendation Models Training**: Building SVD, Item-CF, and Content-Based models.
7.  **Hybrid Recommendation System Development**: Combining models for improved performance.
8.  **Model Testing & Evaluation**: Assessing the accuracy and effectiveness of the recommendation logic.
9.  **Recommendation Display & Visualization**: Presenting recommendations with course names and scores.
10. **Save Trained Models**: Storing the trained models for deployment or later use.

## Technologies Used

*   **Python**: Primary programming language.
*   **Pandas**: Data manipulation and analysis.
*   **NumPy**: Numerical operations.
*   **Matplotlib / Seaborn**: Data visualization.
*   **Scikit-learn**: Feature extraction and preprocessing (TF-IDF, OrdinalEncoder).
*   **Surprise**: A Python scikit for building and analyzing recommender systems.
*   **Joblib**: For saving and loading Python objects (trained models).

## Setup and Installation

To run this project, you'll need a Python environment and the required libraries. It's recommended to use a virtual environment.

1.  **Clone the repository (or download the files):**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise joblib openpyxl
    ```
    *Note: The project specifically requires `numpy<2` for compatibility with `scikit-surprise`. If you encounter issues, try: `pip uninstall numpy && pip install 'numpy<2'`*

## Dataset

The dataset used for this project is `online_course_recommendation.xlsx`. It should be placed in the same directory as the notebook or its path should be updated in the code.

## How to Run the Project

1.  **Open the Jupyter Notebook/Colab**: Open the `online_course_recommendation.ipynb` notebook in Jupyter Lab, Jupyter Notebook, or Google Colab.
2.  **Execute Cells Sequentially**: Run all cells in the notebook from top to bottom. The notebook is designed to be executed sequentially.
3.  **Explore Recommendations**: The final sections of the notebook will demonstrate how to generate and display recommendations for a sample user using the SVD, Item-CF, Content-CF, and Hybrid models.

    Example usage of the recommendation functions:
    ```python
    # Example: Display recommendations for a sample user
    sample_user = df['user_id'].sample(1).iloc[0]
    show_recommendations_with_scores(sample_user, top_n=10)
    ```

## Saved Models

The trained SVD and Hybrid models are saved as `best_svd_model.pkl` and `best_hybrid_model.pkl` respectively. These can be loaded to make predictions without re-training:

```python
import joblib

loaded_svd_model = joblib.load('best_svd_model.pkl')
loaded_hybrid_data = joblib.load('best_hybrid_model.pkl')

# Access components from the hybrid model
# loaded_svd = loaded_hybrid_data['svd']
# loaded_item_cf = loaded_hybrid_data['item_cf']
# loaded_tfidf_matrix = loaded_hybrid_data['tfidf_matrix']
# loaded_course_idx_map = loaded_hybrid_data['course_idx_map']
# loaded_df = loaded_hybrid_data['df']
