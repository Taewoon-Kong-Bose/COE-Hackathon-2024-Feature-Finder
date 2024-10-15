"""
embed_snowflake.py

This script embeds product review records in Snowflake. It queries records 
where embeddings are absent, generates embeddings using a Sentence Transformer model, 
and writes these embeddings back to Snowflake.
"""


import spcs_helpers
from sentence_transformers import SentenceTransformer
import pandas as pd
import json 

def main():
    # Initialize Snowpark session using the helper function
    snowpark_session = spcs_helpers.session()

    # Load data from Snowflake where embedding is NULL
    print("Querying data...")
    query = """
    SELECT DISTINCT _ID, PHRASE
    FROM FEATURE_FINDER
    WHERE EMBEDDING IS NULL
    LIMIT 5000;
    """
    df = snowpark_session.sql(query).toPandas()

    # Load Sentence Transformer model locally
    model = SentenceTransformer('all-mpnet-base-v2')

    # Generate embeddings
    print("Embedding records...")
    df['EMBEDDING'] = df['PHRASE'].apply(lambda x: model.encode(x).tolist() if pd.notna(x) else None)

    total_records = len(df)
    # Iterate through the DataFrame and update each row back into Snowflake
    print("Writing records to Snowflake...")
    for index, row in df.iterrows():
        embedding_json = json.dumps(row['EMBEDDING'])
        # Prepare the SQL update statement with placeholders for parameters
        update_stmt = """
        UPDATE FEATURE_FINDER
        SET EMBEDDING = PARSE_JSON(%s)
        WHERE _ID = %s
        """
        # Execute the update statement using Snowpark session
        snowpark_session.sql(update_stmt, [embedding_json, row['_ID']]).collect()
        # Report progress
        print(f"Processed record {index + 1}/{total_records}.")

if __name__ == "__main__":
    main()