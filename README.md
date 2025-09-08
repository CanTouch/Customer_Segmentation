# Customer_Segmentation
End-to-End Customer Segmentation Analysis
Project Overview
This project provides a complete, end-to-end solution for customer segmentation using transactional data. The goal is to transform raw sales records into actionable business intelligence by identifying distinct customer groups based on their purchasing behavior.

The core of the analysis is the Recency, Frequency, and Monetary (RFM) framework, which distills complex customer behavior into three powerful, interpretable metrics. These metrics are then used for unsupervised machine learning to cluster customers into meaningful segments, or "personas," that marketing and product teams can leverage for targeted campaigns.

Methodology
The project follows a systematic, reproducible data science workflow:

Data Ingestion & Cleaning: Load raw transactional data, handle missing values, and filter out irrelevant transactions (e.g., canceled orders).

Feature Engineering: Calculate RFM metrics for each customer.

Data Preparation: Address the skewed distribution of RFM features using log transformations and standardize the data to prepare it for clustering.

Model Selection: Use the Silhouette Score and Davies-Bouldin Index to determine the optimal number of clusters for the K-Means algorithm.

Clustering & Interpretation: Apply K-Means clustering to segment customers and interpret the resulting clusters by defining clear, descriptive personas (e.g., "Champions," "At-Risk," "Loyal Customers").

Operationalization: Automate the entire analysis pipeline and build an interactive dashboard to make the insights accessible to business stakeholders.

Project Structure
├── data/
│   └── raw/
│       └── online-retail.xlsx  # Raw data file
├── app.py                     # The Streamlit dashboard application
├── customer_segmentation.ipynb  # The Jupyter Notebook containing the full analysis
├── README.md                  # This file
├── requirements.txt           # List of project dependencies
└── Makefile                   # Automates the project workflow

Key Deliverables
Jupyter Notebook (customer_segmentation.ipynb): A comprehensive, step-by-step notebook documenting the entire analysis, from initial data exploration to final persona creation.

Interactive Dashboard (app.py): A production-ready, interactive web application built with Streamlit that allows users to explore and visualize the customer segments in real-time.

How to Run the Project
To run this project, you need to have Python 3.8+ installed. Follow these steps:

Clone the Repository:

git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
cd your-project-name

Install Dependencies:
The project uses a requirements.txt file to manage its dependencies. It's recommended to do this within a virtual environment.

pip install -r requirements.txt

Run the Dashboard:
With all dependencies installed, you can launch the interactive dashboard from your terminal.

streamlit run app.py

This command will automatically open a new browser tab with the running application.
