"""
home.py

The homepage for the semantic search interface.
"""

# Standard library imports
import json

# Third-party library imports
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local application/library specific imports
import spcs_helpers

st.set_page_config(page_title="Review Feature Finder", layout="wide")

@st.cache_resource
def connect_to_snowflake():
    return spcs_helpers.session()

@st.cache_data(ttl=None, show_spinner=True)
def load_and_prepare_data():
    conn = connect_to_snowflake()
    query = """
        SELECT BRAND, NAME, PRODUCTTYPE, RATING, REVIEWDATE, PHRASE, CANDIDATE_LABEL, ENTITY, ASPECT_1, ASPECT_2, SENTIMENT, EMBEDDING
        FROM TAICHI.TNG.V_FEATURE_FINDER
    """
    df = conn.sql(query).toPandas()
    df['EMBEDDING'] = df['EMBEDDING'].apply(json.loads)
    return df

model = SentenceTransformer('all-mpnet-base-v2')

#UI text strings
page_title = "Review Feature Finder"
page_helper = "Explore reviews based on the features you want to know about!"
    #informative component labels   
empty_search_helper = "Enter a search term to proceed."
#empty_search_helper = "Enter a search term and select the filters you would like to apply"
semantic_search_header = "What feature are you looking for?"
semantic_search_placeholder = "spatial audio"
brand_filter_header = "Brand"
product_filter_header = "Product Name"
product_type_filter_header = "Product Type"
rating_filter_header = "Rating"
sentiment_filter_header = "Sentiment"
start_date_filter_header = "Ratings After"
end_date_filter_header = "Rating Before"
    #results text   
table_header = "Reviews found based on feature"
graph_header = "T-SNE Visualization of Phrase Embeddings with Search Query" 
graph2_header = "" #TODO: Pratik to add
    #button labels
search_label = "Search Reviews"
reset_label = "Reset Search"
set_filters_label = "Set Filters"
reset_filters_label = "Reset Filters"
download_label = "Download Results" 
visualize_data_label = "Visualize Data"
st.session_state.search_reset_disabled = True
st.session_state.search_disabled = True     #user can't click button until value is entered

#! this is a hacky workaround
# TODO reorder the streamlit elements to load the page in a more elegant way
st.session_state.df = None

def handler_search_features():
    """
    Fetch product review records from Snowflake and filter them based on 
    the cosine similarity of their embeddings to a user-provided input embedding.
    """

    if "user_input" in st.session_state and st.session_state.user_input != "":
    
        # Encode user input to embedding
        user_input_embed = model.encode(st.session_state.user_input).reshape(1, -1)
        st.session_state.user_input_embed = user_input_embed
        df = st.session_state.df
        # Compute cosine similarity between fetched embeddings and user input embedding
        df['cosine_similarity'] = df['EMBEDDING'].apply(
            lambda emb: cosine_similarity([emb], user_input_embed)[0][0]
        )
        # Filter, sort, and clean up the DataFrame
        user_input_df = df[df['cosine_similarity'] > 0.5]
        user_input_df = user_input_df.sort_values(by='cosine_similarity', ascending=False)
        #user_input_df = user_input_df.drop(columns=['EMBEDDING', 'cosine_similarity']).reset_index(drop=True)
        # Store results in session state
        st.session_state.user_input_df = user_input_df
        st.session_state.fltrd_df = user_input_df

    else:
        if 'user_input_df' in st.session_state:
            del st.session_state['user_input_df']


def handler_reset_search():
    if "user_input" in st.session_state and st.session_state.user_input != "":
        st.session_state.user_input = ""
        st.session_state.search_disabled = False
        st.session_state.search_reset_disabled = False
        if 'user_input_df' in st.session_state:
            del st.session_state['user_input_df']


def handler_filter_results():
    st.session_state.fltrd_df = pd.DataFrame()

    if "brands_selection" in st.session_state:
        st.session_state.fltrd_df = pd.concat([st.session_state.fltrd_df, st.session_state.user_input_df[st.session_state.user_input_df['BRAND'].isin(st.session_state.brands_selection)]])
 
    if "products_selection" in st.session_state:
        st.session_state.fltrd_df = pd.concat([st.session_state.fltrd_df, st.session_state.user_input_df[st.session_state.user_input_df['NAME'].isin(st.session_state.products_selection)]])
    
    if "product_type_selection" in st.session_state:
        st.session_state.fltrd_df = pd.concat([st.session_state.fltrd_df, st.session_state.user_input_df[st.session_state.user_input_df['PRODUCTTYPE'].isin(st.session_state.product_type_selection)]])

    if "rating_selection" in st.session_state:
        st.session_state.fltrd_df = pd.concat([st.session_state.fltrd_df, st.session_state.user_input_df[st.session_state.user_input_df['RATING'].isin(st.session_state.rating_selection)]])
    
    if "sentiment_selection" in st.session_state:
        st.session_state.fltrd_df = pd.concat([st.session_state.fltrd_df, st.session_state.user_input_df[st.session_state.user_input_df['SENTIMENT'].isin(st.session_state.sentiment_selection)]])

def handler_reset_filters():
    #resets filters on search output
    st.session_state.brands_selection = []
    st.session_state.products_selection = []
    st.session_state.product_type_selection = []
    st.session_state.rating_selection = []
    st.session_state.sentiment_selection = []
    st.session_state.fltrd_df = st.session_state.user_input_df

def download_data_to_csv():
    #Convert DataFrame to CSV and then to bytes

    if "fltrd_df" in st.session_state and st.session_state.fltrd_df.size>0:
        csv = st.session_state.fltrd_df.to_csv(index=False).encode('utf-8')
    else:
        csv = st.session_state.user_input_df.to_csv(index=False).encode('utf-8')


    if "user_input" in st.session_state and st.session_state.user_input != "":
        st.download_button(
            label=download_label,
            data=csv,
            file_name="sample_data.csv",
            mime='text/csv',
        )

def filter_categories():
    #creates values for filters based on embedding table state
    if "user_input_df" in st.session_state:
        st.session_state.brand_categories = sorted(st.session_state.user_input_df['BRAND'].unique())
        st.session_state.name = sorted(st.session_state.user_input_df['NAME'].unique())
        st.session_state.product_type_categories = sorted(st.session_state.user_input_df['PRODUCTTYPE'].unique())
        st.session_state.rating_categories = sorted(st.session_state.user_input_df['RATING'].unique())
        st.session_state.sentiment_categories = sorted(st.session_state.user_input_df['SENTIMENT'].unique())
        st.session_state.start_date = pd.to_datetime(st.session_state.user_input_df['REVIEWDATE'].min())
        st.session_state.end_date = pd.to_datetime(st.session_state.user_input_df['REVIEWDATE'].max())

#Search Components 
def render_search():
    #Renders search form in expander at top of page 

    #user can't click button until value is entered
    st.session_state.search_disabled=True

    expander = st.expander("Search for a feature: ", expanded=True)
    with expander:  

        #feature search box
        st.text_input(label=semantic_search_header, placeholder = semantic_search_placeholder, key = "user_input", on_change = handler_search_features)
        if "user_input" in st.session_state and st.session_state.user_input != "":
            st.session_state.search_disabled = False
            st.session_state.search_reset_disabled = False
        else:
            st.session_state.search_disabled = True
            st.session_state.search_reset_disabled = True
             

        #TODO:click action when functions are created
        colz, cola, colb = st.columns(3)
        with cola:
            st.button(label=search_label, key = "feature_search", disabled = st.session_state.search_disabled, on_click = handler_search_features)
        with colb:
            st.button(label=reset_label, key = "reset_search", disabled = st.session_state.search_reset_disabled, on_click = handler_reset_search)

def render_search_result():
    #displays table result of embeddings

    #TODO remove, Miranda using for filter development
    if "fltrd_df" in st.session_state and st.session_state.fltrd_df.size>0:
        st.write(st.session_state.fltrd_df)
    else:
        st.write(st.session_state.user_input_df)
    ####

#Filter Components
def filters():

    # col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
            st.multiselect(label = brand_filter_header, options = st.session_state.brand_categories, default = [], key = "brands_selection")
    with col2:
            st.multiselect(label = product_filter_header, options = st.session_state.name, default = [], key = "products_selection")
    with col3:
            st.multiselect(label = product_type_filter_header, options = st.session_state.product_type_categories, default = [], key = "product_type_selection")
    with col4:
            st.multiselect(label = rating_filter_header, options = st.session_state.rating_categories, default = [], key = "rating_selection")
    with col5:
            st.multiselect(label = sentiment_filter_header, options = st.session_state.sentiment_categories, default = [], key = "sentiment_selection")
    # with col6:
    #         st.date_input(label = start_date_filter_header)
            # st.date_input(label = start_date_filter_header, min_value = st.session_state.start_date, max_value = st.session_state.end_date)
    # with col7:
    #         st.date_input(label = end_date_filter_header)
            # st.date_input(label = end_date_filter_header, min_value = st.session_state.start_date, max_value = st.session_state.end_date)
 
    coly, colz = st.columns(2)
    with coly:
        st.button(label=set_filters_label, key = "set_filters", on_click = handler_filter_results)
    with colz:
        st.button(label=reset_filters_label, key = "reset_filters", on_click = handler_reset_filters)

def calculate_perplexity(n_records):
    """
    Calculate the perplexity based on the number of records.
    Adjust the factor to scale perplexity appropriately for your specific dataset.
    """
    #if n_records < 30:
     #   return 5  # Set a minimum value for small datasets
    return min(50, max(1, np.sqrt(n_records) * 0.1))  # Scale between 5 and 50 based on sqrt of dataset size

def calculate_cluster_size(n_records):
    return int(np.sqrt(n_records / 2))


def generate_embedding_clusters():
    
    df = st.session_state.user_input_df
    if len(df)>1:
        embeddings = df['EMBEDDING']

        n_records = len(df)
        perplexity = calculate_perplexity(n_records)

        # Convert Series of lists to a 2D numpy array
        search_query_embedding = np.array(st.session_state.user_input_embed)  # Replace this with the actual search query embedding
        embeddings_array = np.stack(embeddings.values)
        combined_embeddings = np.vstack([embeddings_array, search_query_embedding])

        # Apply t-SNE to the combined dataset
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, random_state=42)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)

        # Clustering (optional for the original embeddings)
        cluster_size = calculate_cluster_size(n_records)
        kmeans = KMeans(n_clusters=cluster_size)
        clusters = kmeans.fit_predict(reduced_embeddings[:-1])  # Exclude the search query from clustering

        reviews = df['PHRASE']
            
        # Create a scatter plot for the dataset
        fig = go.Figure()

        # Add the clusters as scatter plot
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[:-1, 0],
            y=reduced_embeddings[:-1, 1],
            mode='markers',
            marker=dict(color=clusters, size=8, colorscale='Viridis', showscale=True),
            text=reviews[:-1],  # Exclude the search query from the cluster labels
            hoverinfo='text',  # Display the text on hover
            name='Clusters'
        ))

        # Add the search query as a distinct red dot
        fig.add_trace(go.Scatter(
            x=[reduced_embeddings[-1, 0]],
            y=[reduced_embeddings[-1, 1]],
            mode='markers',
            marker=dict(color='red', size=10),
            text=st.session_state.user_input,  # Search query text
            hoverinfo='text',  # Display the text on hover
            name='Search Query'
        ))

        # Update plot layout
        fig.update_layout(
            title=graph_header,
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            legend_title='Legend'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, height=600)

def dynamic_histograms(category):
    df = st.session_state.user_input_df

    if category == 'BRAND':
         # Creating the histogram using Plotly Express
        fig = px.histogram(df, x='BRAND', nbins=5, title='Distribution of Brands')

        # Adding histogram customization
        fig.update_layout(
            xaxis_title='Brand',
            yaxis_title='Count',
            bargap=0.2,  # gap between bars of adjacent location coordinates
        )

        st.plotly_chart(fig)

    elif category == 'RATING':
         # Creating the histogram using Plotly Express
        fig = px.histogram(df, x='RATING', nbins=5, title='Distribution of Ratings')

        # Adding histogram customization
        fig.update_layout(
            xaxis_title='Rating',
            yaxis_title='Count',
            bargap=0.2,  # gap between bars of adjacent location coordinates
        )

        st.plotly_chart(fig)
    
    elif category == 'SENTIMENT':
         # Creating the histogram using Plotly Express
        fig = px.histogram(df, x='SENTIMENT', nbins=5, title='Distribution of Sentiment')

        # Adding histogram customization
        fig.update_layout(
            xaxis_title='Sentiment',
            yaxis_title='Count',
            bargap=0.2,  # gap between bars of adjacent location coordinates
        )

        st.plotly_chart(fig)

    else:
         # Creating the histogram using Plotly Express
        fig = px.histogram(df, x='CANDIDATE_LABEL', nbins=5, title='Distribution of CANDIDATE_LABEL')

        # Adding histogram customization
        fig.update_layout(
            xaxis_title='CANDIDATE_LABEL',
            yaxis_title='Count',
            bargap=0.2,  # gap between bars of adjacent location coordinates
        )
        st.plotly_chart(fig)

#Page Layout
st.title(page_title)
st.write(page_helper)
st.write("---")
render_search()

st.session_state.df = load_and_prepare_data()


if "user_input_df" not in st.session_state:
    st.write(empty_search_helper)

else:
    df = st.session_state.user_input_df
    if len(df)>1:
        filter_categories()
        filters()
        render_search_result()
        download_data_to_csv()
        col1, col2, = st.columns(2)

        with col1:
            generate_embedding_clusters()

        with col2:
            st.markdown('**Histograms**')
            columns_list = ['BRAND', 'RATING', 'SENTIMENT', 'CANDIDATE_LABEL']
            category = st.selectbox('Select a Category', columns_list)
            dynamic_histograms(category)