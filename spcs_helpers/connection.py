"""
connection.py

Establish a connection to Snowflake using the details in .streamlit/secrets.toml.
"""

# Standard library imports
import toml

# Related third-party imports
import snowflake.connector
from snowflake.snowpark import Session

def load_config():
    """
    Load database configuration from secrets.toml.

    Returns
    -------
    dict
        A dictionary containing database connection parameters.
    """
    # Load the configuration from secrets.toml
    config = toml.load(".streamlit/secrets.toml")
    return config['snowflake']

def connection() -> snowflake.connector.SnowflakeConnection:
    """
    Establish a connection to a Snowflake database for a local environment.

    This function creates a connection to a Snowflake database using user/password authentication
    with an external browser for authentication.

    Returns
    -------
    snowflake.connector.SnowflakeConnection
        An established Snowflake connection object.
    """
    # Load credentials from configuration
    creds = load_config()
    
    # Adjust the credentials for local setup using an external browser
    creds.update({
        'authenticator': "EXTERNALBROWSER",
        'client_session_keep_alive': True
    })
    # Establish and return the Snowflake connection
    return snowflake.connector.connect(**creds)

def session() -> Session:
    """
    Initialize and return a Snowpark session for a local environment.

    Returns
    -------
    Session
        The initialized Snowpark session.
    """
    # Load all configurations for the session
    config = load_config()
    
    conn = connection()
    return Session.builder.configs({
        "connection": conn,
        **config  # Unpacking all additional configs loaded from TOML
    }).create()
