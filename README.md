# Recommendation-System-for-Nike
This project develops a personalized product recommendation system for Nike Plus, focusing on customer preferences like size and color. By leveraging content-based filtering, cosine similarity, and TF-IDF vectorization, it delivers tailored recommendations to enhance user shopping experiences with highly relevant product suggestions.

Features:
Content-Based Filtering: Generates recommendations by analyzing product attributes, including size and color, to match user preferences.
Cosine Similarity & TF-IDF: Converts and compares product attributes for similarity calculations to provide relevant suggestions.
Interactive Streamlit Web App: Allows users to input preferences and receive personalized recommendations in real-time.
Tableau Dashboard: Offers visual insights into product performance and key metrics.

DatasetDescription:
https://www.kaggle.com/datasets/polartech/nike-sportwear-product-dataset
The dataset contains key attributes of Nike Plus products, including department, category, subcategory, size, color, price, and product type. This data is used to personalize recommendations, focusing on the most relevant features for enhancing user engagement.

Methodology:
Data Preparation: Cleaning, encoding, and normalizing product data for effective feature representation.
Feature Engineering: Vectorizing textual data (e.g., colors) using TF-IDF and encoding categorical features (e.g., size, type).
Model Development: Content-based recommendation using cosine similarity to identify and suggest similar products.
Deployment: Streamlit app for user interaction and real-time recommendations.
Results
The system achieved high precision in recommendations, ensuring accuracy and relevance based on user input preferences. The integration of a feedback loop further refines the recommendations over time.

Future Work:
Hybrid Approach: Integrating collaborative filtering with content-based methods for improved recommendations.
Enhanced NLP Techniques: Applying advanced models to better understand and incorporate product features.
User Feedback: Refining the system based on continuous user interactions and feedback.

Technologies Used:
Python (Pandas, NumPy, Sklearn)
Streamlit for web app deployment
Jupyter Notebook for model development
Tableau for data visualization
