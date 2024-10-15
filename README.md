# COE-Hackathon-2024-Feature-Finder

NOTE: This README is a WIP, please edit as necessary!

This prototype features a streamlit app that enables users to query product reviews for direct and indirect mentions of product features. For instance, when searching for all references to "spatial audio" it will return mentions like “spacialized audio” (sp), “multidimensional sound”, “spatial sound feature”, “the spacialization of sound”, “spatial enhancement”, and others.

## Setup
To run the prototype locally you have to
1. Clone this repo to your machine:
    ```bash
    git clone https://github.com/BoseCorp/COE-Hackathon-2024-Feature-Finder.git
    ```
1. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
1. Activate the virtual environment

    - On Windows run: 
        ```bash 
        .\venv\Scripts\activate
        ```

    - On MacOS or Linux run:
        ```bash
        source venv/bin/activate
        ```
1. Install the dependencies in requirements.txt:
    ```bash
    pip install -r requirements.txt
    ```

1. Run `streamlit run home.py`

1. Configure `secrets.toml` file:
    - If the `.streamlit` folder does not exist in your project directory, create it:
        ```bash
        mkdir .streamlit
        ```
    - Inside the `.streamlit` folder, create a file named `secrets.toml`.
    - You can create this file using a text editor or by running the following command in the terminal:
        ```bash
        touch .streamlit/secrets.toml
        ```
    - Open `secrets.toml` in your preferred text editor & add the necessary secret keys & values in the TOML format:
        ```
        [snowflake]
        account = "bose.us-east-1"
        user = "YOUR_SNOWFLAKE_USER"
        role = "YOUR_SNOWFLAKE_ROLE"
        warehouse = "AID_TAICHI_XSMALL_WH"
        database = "TAICHI"
        schema = "TNG"
        ```