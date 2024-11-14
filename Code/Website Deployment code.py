import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = r"D:\Dissertation\Nike_UK_2022-09-01.csv"
data = pd.read_csv(file_path)

# Extract relevant columns for the recommendation system
relevant_data = data[['PRODUCT_NAME', 'PRODUCT_SIZE', 'COLOR', 'PRODUCT_TYPE', 'PRICE_CURRENT', 'PRODUCT_URL', 'DEPARTMENT']]

# Handling 'PRODUCT_SIZE' - convert sizes to numerical representations
size_encoder = LabelEncoder()
relevant_data['SIZE_ENCODED'] = size_encoder.fit_transform(relevant_data['PRODUCT_SIZE'])

# Handling 'COLOR' - vectorize the color names
color_vectorizer = CountVectorizer()
color_vectors = color_vectorizer.fit_transform(relevant_data['COLOR']).toarray()

# Handling 'PRODUCT_TYPE' - convert product types to numerical representations
product_type_encoder = LabelEncoder()
relevant_data['TYPE_ENCODED'] = product_type_encoder.fit_transform(relevant_data['PRODUCT_TYPE'])

# Handling 'PRICE_CURRENT' - normalize prices
scaler = StandardScaler()
relevant_data['PRICE_NORMALIZED'] = scaler.fit_transform(relevant_data[['PRICE_CURRENT']])

# Combine all features into a single feature vector
size_scaled = scaler.fit_transform(relevant_data[['SIZE_ENCODED']])
additional_features = relevant_data[['TYPE_ENCODED', 'PRICE_NORMALIZED']]
additional_feature_vectors = scaler.fit_transform(additional_features)
feature_vectors = np.hstack((size_scaled, color_vectors, additional_feature_vectors))

# Get unique categories and departments from the dataset
unique_categories = relevant_data['PRODUCT_TYPE'].unique()
unique_departments = relevant_data['DEPARTMENT'].unique()

def find_product_index(product_name, category, size, color, department, data):
    # Filter the data based on the input criteria
    matches = data[
        (data['PRODUCT_NAME'].str.contains(product_name, case=False, na=False)) &
        (data['PRODUCT_TYPE'].str.contains(category, case=False, na=False)) &
        (data['PRODUCT_SIZE'].str.contains(size, case=False, na=False)) &
        (data['DEPARTMENT'].str.contains(department, case=False, na=False))
    ]
    
    # Only filter by color if a specific color is provided
    if color is not None:
        matches = matches[matches['COLOR'].str.contains(color, case=False, na=False)]
    
    if not matches.empty:
        return matches.index[0]  # Return the index of the first match
    else:
        return None  # Return None if no match is found

# Function to get recommendations based on product index
def get_recommendations(product_index, feature_vectors, relevant_data, top_n=5):
    # Calculate cosine similarity between products
    similarities = cosine_similarity([feature_vectors[product_index]], feature_vectors)[0]
    # Get indices of top similar products
    similar_indices = similarities.argsort()[::-1][1:]  # Exclude the queried product itself
    # Exclude duplicate product names from the recommendations
    seen_products = set()
    unique_recommendations = []
    for index in similar_indices:
        product_name = relevant_data.iloc[index]['PRODUCT_NAME']
        if product_name not in seen_products:
            seen_products.add(product_name)
            unique_recommendations.append(index)
        if len(unique_recommendations) >= top_n:
            break
    # Get details of the top unique similar products
    recommended_products = relevant_data.iloc[unique_recommendations]
    return recommended_products

# Streamlit App
st.set_page_config(page_title="Nike Product Recommender", page_icon=":athletic_shoe:", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #fff;
            font-family: Arial, sans-serif;
        }
        h1 {
            color: #111;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }
        h2 {
            color: #111;
            font-size: 24px;
            font-weight: bold;
        }
        .header {
            background-color: #111;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .recommendation-card {
            border-radius: 15px;
            background-color: #f5f5f5;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        .recommendation-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .recommendation-info {
            font-size: 18px;
            margin-bottom: 5px;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        a {
            color: #111;
            font-weight: bold;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS Styling for the Navigation Bar
st.markdown("""
    <style>
        .navbar {
            background-color: white;
            padding: 10px 20px;
            border-bottom: 1px solid #eaeaea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar-left {
            display: flex;
            align-items: center;
        }
        .navbar-left img {
            margin-right: 20px;
        }
        .navbar-right a {
            color: black;
            margin-right: 20px;
            text-decoration: none;
            font-weight: bold;
        }
        .navbar-right a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Correct paths to the uploaded images
jordan_logo_path = "/mnt/data/file-mhLOdZoudyW90yrRAVqjYCiU"  # This is the file ID for the Jordan logo
converse_logo_path = "/mnt/data/image.png"  # Assuming this is the Converse logo you uploaded earlier

# Custom CSS Styling for the Navigation Bar
st.markdown("""
    <style>
        .navbar {
            background-color: white;
            padding: 10px 20px;
            border-bottom: 1px solid #eaeaea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar-left {
            display: flex;
            align-items: center;
        }
        .navbar-left img {
            margin-right: 20px;
        }
        .navbar-right a {
            color: black;
            margin-right: 20px;
            text-decoration: none;
            font-weight: bold;
        }
        .navbar-right a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS Styling for the Navigation Bar
st.markdown("""
    <style>
        .navbar {
            background-color: white;
            padding: 10px 20px;
            border-bottom: 1px solid #eaeaea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar-left {
            display: flex;
            align-items: center;
        }
        .navbar-right {
            display: flex;
            justify-content: flex-end;
        }
        .navbar-right a {
            color: black;
            margin-right: 20px;
            text-decoration: none;
            font-weight: bold;
        }
        .navbar-right a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Render the navigation bar without images
st.markdown('<div class="navbar">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])  # Adjust the width ratio of columns if needed

with col1:
    st.markdown("")  # Left column is empty (no images)

with col2:
    st.markdown("""
        <div class="navbar-right">
            <a href="#">Find a Store</a>
            <a href="#">Help</a>
            <a href="#">Join Us</a>
            <a href="#">Sign In</a>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Nike logo at the top
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" alt="Nike Logo" width="200">
    </div>
    """, unsafe_allow_html=True
)

st.title("Nike Product Recommendation System")

st.markdown("""
    <div class="header" style="background-color: white; color: blue; padding: 10px; border-radius: 10px; text-align: center;">
        <h2>Find Your Perfect Nike Product</h2>
    </div>
""", unsafe_allow_html=True)


# Create columns for horizontal input fields
col1, col2, col3, col4, col5 = st.columns(5)

# Place the input fields horizontally
with col1:
    user_input_department = st.selectbox("Department", unique_departments)

with col2:
    user_input_name = st.text_input("Product Name")

with col3:
    user_input_category = st.selectbox("Category", unique_categories)

with col4:
    if user_input_category.lower() == 'footwear':
        user_input_size = st.text_input("Size (e.g., 8, 9, 10)")
    elif user_input_category.lower() == 'apparel':
        user_input_size = st.text_input("Size (e.g., S, M, L)")
    else:
        user_input_size = st.text_input("Size")

with col5:
    user_input_color = st.text_input("Color (e.g., Red, Black)")


# Adjust search criteria based on whether the user selected "Any" for the color
if user_input_color.lower() == 'any':
    product_index = find_product_index(user_input_name, user_input_category, user_input_size, None, user_input_department, relevant_data)
else:
    product_index = find_product_index(user_input_name, user_input_category, user_input_size, user_input_color, user_input_department, relevant_data)

# Find the product index based on user input
if product_index is not None:
    # Get recommendations for the found product
    recommended_products = get_recommendations(product_index, feature_vectors, relevant_data)

    st.subheader(f"Recommendations for {relevant_data.iloc[product_index]['PRODUCT_NAME']} in {user_input_department} Department")

   # Display recommended products in horizontal layout using dynamic columns
    num_recommendations = len(recommended_products)
    if num_recommendations > 0:
        # Limit to a maximum of 5 columns for better UI control
        cols_per_row = min(num_recommendations, 5)
        
        rows = (num_recommendations // cols_per_row) + (1 if num_recommendations % cols_per_row else 0)
        
        # Display the products row by row
        for row_idx in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row_idx * cols_per_row + col_idx
                if idx < num_recommendations:
                    product = recommended_products.iloc[idx]
                    with cols[col_idx]:
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="recommendation-title">{product['PRODUCT_NAME']}</div>
                                <div class="recommendation-info">Color: {product['COLOR']}</div>
                                <div class="recommendation-info">Type: {product['PRODUCT_TYPE']}</div>
                                <div class="recommendation-info">Price: £{product['PRICE_CURRENT']}</div>
                                <div class="recommendation-info"><a href="{product['PRODUCT_URL']}" target="_blank">View on Nike Website</a></div>
                            </div>
                        """, unsafe_allow_html=True) 
    else:
        st.write("No recommendations available.")
else:
    st.error("Product not found or no matching products available. Please try again with different criteria.")