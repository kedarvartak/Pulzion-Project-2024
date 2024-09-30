import streamlit as st
import pandas as pd
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import pymysql
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import wavio
from pymongo import MongoClient
from bson import ObjectId
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3307")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "kedar")
DB_NAME = os.getenv("DB_NAME", "friends")
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="💬 Chat with MySQL/CSV/MongoDB",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the ChatGroq model
chat_groq = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, temperature=0)

# Initialize the SQL Database connections
def init_databases(user: str, password: str, host: str, port: str, database: str) -> tuple[SQLDatabase, SQLDatabase]:
    try:
        db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        query_db = SQLDatabase.from_uri(db_uri)
        schema_db = SQLDatabase.from_uri(db_uri)
        logger.info("MySQL connection established successfully.")
        return query_db, schema_db
    except Exception as e:
        logger.error(f"MySQL Connection Error: {e}")
        raise e

# Initialize MongoDB client
def init_mongo_db(uri: str, dbname: str):
    try:
        client = MongoClient(uri)
        db = client[dbname]
        logger.info("MongoDB connection established successfully.")
        return db
    except Exception as e:
        st.error(f"❌ Error connecting to MongoDB: {e}")
        logger.error(f"MongoDB Connection Error: {e}")
        return None

# Get MongoDB collection
def get_mongo_collection(db, collection_name):
    try:
        # Ensure that the collection name is valid (not empty)
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        collection = db[collection_name]
        # Test the collection by counting documents
        if collection.count_documents({}) is not None:
            logger.info(f"MongoDB collection '{collection_name}' accessed successfully.")
            return collection
    except Exception as e:
        st.error(f"❌ Error accessing collection '{collection_name}': {e}")
        logger.error(f"MongoDB Collection Access Error: {e}")
        return None

# Sidebar: CSV upload feature
def load_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        st.session_state.csv_data = df
        st.success(f"✅ Successfully uploaded: {file.name}")
        logger.info(f"CSV file '{file.name}' loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")
        logger.error(f"CSV Loading Error: {e}")

# Display schema of uploaded CSV (columns and their types)
def display_csv_schema():
    if "csv_data" in st.session_state and st.session_state.csv_data is not None and not st.session_state.csv_data.empty:
        df = st.session_state.csv_data
        st.write("*CSV Schema:*")
        st.json({col: str(df[col].dtype) for col in df.columns})
        logger.info("CSV schema displayed successfully.")
    else:
        st.warning("⚠️ No CSV uploaded. Please upload a CSV file first.")
        logger.warning("Attempted to display CSV schema without uploading a CSV.")

# Sanitize SQL queries by removing unwanted backslashes before underscores
def sanitize_query(query: str) -> str:
    sanitized = query.replace('\\', '')
    logger.debug(f"Sanitized SQL query: {sanitized}")
    return sanitized

# Function to parse and generate user-friendly error messages
def parse_error_message(exception: Exception) -> str:
    if isinstance(exception, pymysql.err.OperationalError):
        error_code = exception.args[0]
        error_msg = exception.args[1]
        if error_code == 1054:
            match = re.search(r"Unknown column '(.+?)' in 'field list'", error_msg)
            if match:
                column = match.group(1)
                return f"❌ *Error:* The column '{column}' does not exist in the specified table. Please check your query and try again."
            return "❌ *Error:* There was an issue with your query. Please verify the column names and try again."
        elif error_code == 1146:
            match = re.search(r"Table '(.+?)' doesn't exist", error_msg)
            if match:
                table = match.group(1)
                return f"❌ *Error:* The table '{table}' does not exist in the database. Please check your query and try again."
            return "❌ *Error:* There was an issue with your query. Please verify the table names and try again."
        else:
            return "❌ *Error:* There was an issue executing your query. Please ensure it's correct and try again."
    elif isinstance(exception, pymysql.err.ProgrammingError):
        return "❌ *Error:* There was a syntax error in your query. Please review it and try again."
    else:
        return "❌ *Error:* An unexpected error occurred while processing your request. Please try again later."

# Get SQL chain for the model interaction (Queries)
def get_sql_chain(query_db: SQLDatabase, schema_db: SQLDatabase):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question.

    <SCHEMA>{schema}</SCHEMA>

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    Do not escape any characters in the SQL query.

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = chat_groq

    def get_schema(_):
        try:
            tables = schema_db.run("SHOW TABLES;")
            schema_info = ""
            for table in tables.split('\n'):
                table = table.strip()
                if table:
                    create_stmt = schema_db.run(f"SHOW CREATE TABLE `{table}`;")
                    schema_info += create_stmt + "\n\n"
            logger.info("Schema retrieved successfully for SQL queries.")
            return schema_info
        except Exception as e:
            logger.error(f"Schema Retrieval Error: {e}")
            return ""

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.invoke
        | StrOutputParser()
    )

# Generate the SQL response based on the user query and database state
def get_response(user_query: str, query_db: SQLDatabase, schema_db: SQLDatabase):
    sql_chain = get_sql_chain(query_db, schema_db)

    try:
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, the question, the SQL query, and the SQL response, write a natural language response.

        <SCHEMA>{schema}</SCHEMA>

        SQL Query: <SQL>{query}</SQL>
        User Question: {question}
        SQL Response: {response}

        Natural Language Response:
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = chat_groq

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: schema_db.run("SHOW TABLES;"),
                response=lambda vars: query_db.run(sanitize_query(vars["query"])),
            )
            | prompt
            | llm.invoke
            | StrOutputParser()
        )

        response = chain.invoke({
            "question": user_query,
        })
        logger.info("SQL response generated successfully.")
        return response
    except Exception as e:
        logger.error(f"SQL Response Error: {e}")
        user_friendly_message = parse_error_message(e)
        return user_friendly_message

# Generate responses based on CSV data
def get_response_csv(user_query: str, df: pd.DataFrame):
    st.write("### Debugging get_response_csv")
    st.write(f"User Query: {user_query}")
    st.write(f"DataFrame Columns: {df.columns.tolist()}")
    st.write(f"DataFrame Head:\n{df.head()}")

    try:
        # Define a more flexible pattern to capture different query formats
        # Example queries:
        # "Provide me the Age where Name = John"
        # "Show me the Salary where Department = Sales"
        # "Get all entries where Country = USA"
        
        pattern = re.compile(
            r"(?i)(?:provide me|get|show me|find|list)\s+(?:the\s+)?(?:entries|rows|data|columns)?\s*(?:of\s+)?(?:column\s+)?(?P<column>\w+)?\s*(?:where|with)\s+(?P<condition_col>\w+)\s*=\s*(?P<condition_val>.+)"
        )
        match = pattern.search(user_query)
        if match:
            column = match.group("column")
            condition_col = match.group("condition_col")
            condition_val = match.group("condition_val").strip().strip("'\"")  # Remove possible quotes

            # Validate columns
            if condition_col not in df.columns:
                return f"❌ The column '{condition_col}' does not exist in the CSV data."
            if column and column not in df.columns:
                return f"❌ The column '{column}' does not exist in the CSV data."

            # Determine the data type of the condition column
            dtype = df[condition_col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                try:
                    condition_val = float(condition_val) if '.' in condition_val else int(condition_val)
                except ValueError:
                    return f"❌ Invalid value type for column '{condition_col}'. Expected a numeric value."
            else:
                condition_val = condition_val  # Keep as string

            # Perform the query
            if column:
                result = df.loc[df[condition_col] == condition_val, column]
                if result.empty:
                    return f"❌ No records found where '{condition_col}' = {condition_val}."
                else:
                    return f"✅ Found value(s) in '{column}': {result.tolist()}"
            else:
                # If no specific column is mentioned, return entire rows
                result = df.loc[df[condition_col] == condition_val]
                if result.empty:
                    return f"❌ No records found where '{condition_col}' = {condition_val}."
                else:
                    # Convert DataFrame to JSON for readability
                    result_json = result.to_json(orient='records')
                    return f"✅ Found {len(result)} record(s):\n{result_json}"
        else:
            return "❌ Unable to understand the query. Please ask in the format: 'provide me <column> where <condition_column> = <value>'."
    except Exception as e:
        logger.error(f"CSV Query Error: {e}")
        return f"❌ Error processing query: {e}"

# Generate MongoDB response based on natural language query using the language model
def get_response_mongo(user_query: str, collection):
    try:
        # Define the prompt to translate natural language to MongoDB query
        prompt = f"""
        Translate the following natural language query into a MongoDB query for the 'movies' collection in the 'sample_mflix' database.
        The response should be a valid MongoDB query in JSON format. Only provide the MongoDB query and nothing else.

        Natural Language Query: "{user_query}"

        MongoDB Query:
        """

        # Use the language model to generate the MongoDB query
        ai_response = chat_groq.invoke(prompt)

        # Extract text content from AIMessage object
        if hasattr(ai_response, 'content'):
            mongo_query = ai_response.content.strip()
        elif hasattr(ai_response, 'text'):
            mongo_query = ai_response.text.strip()
        else:
            return f"❌ Unable to generate a valid MongoDB query from the prompt."

        logger.info(f"Generated MongoDB Query: {mongo_query}")

        # Validate that the response is a valid JSON-like MongoDB query
        if not (mongo_query.startswith("{") and mongo_query.endswith("}")):
            return f"❌ Unable to generate a valid MongoDB query from the prompt."

        # Parse the JSON safely
        try:
            mongo_query_dict = json.loads(mongo_query)
        except json.JSONDecodeError:
            return f"❌ Unable to parse the generated MongoDB query. Ensure it's valid JSON."

        # Execute the generated MongoDB query
        results = collection.find(mongo_query_dict).limit(10)  # Limit to 10 for now
        result_list = list(results)

        if result_list:
            response = f"✅ Found {len(result_list)} document(s):\n"
            for doc in result_list:
                # Convert ObjectId to string for readability
                doc_str = {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in doc.items()}
                response += f"{doc_str}\n"
            logger.info("MongoDB query executed successfully.")
            return response
        else:
            return "❌ No documents found matching your query."
    except Exception as e:
        logger.error(f"MongoDB Query Execution Error: {e}")
        return f"❌ Error processing MongoDB query: {e}"

# Function to reset chat history
def reset_chat_history():
    st.session_state.messages = []
    logger.info("Chat history has been reset due to data source change.")

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process user query (either text or transcription)
def process_query(user_query: str):
    if not isinstance(user_query, str) or user_query.strip() == "":
        st.error("❌ *Error:* Invalid input. Please enter a valid message.")
        return

    user_query = user_query.strip()

    # Append user message to session state
    st.session_state.messages.append({"role": "Human", "content": user_query})
    logger.info(f"User query appended to session state: {user_query}")

    # Determine the response based on the connected data source
    if ("mongo_db" in st.session_state and st.session_state.mongo_db is not None and
        "mongo_collection" in st.session_state and st.session_state.mongo_collection is not None):
        response = get_response_mongo(user_query, st.session_state.mongo_collection)
    elif ("query_db" in st.session_state and "schema_db" in st.session_state and
          st.session_state.query_db is not None and st.session_state.schema_db is not None):
        response = get_response(user_query, st.session_state.query_db, st.session_state.schema_db)
    elif ("csv_data" in st.session_state and st.session_state.csv_data is not None and
          not st.session_state.csv_data.empty):
        df = st.session_state.csv_data
        response = get_response_csv(user_query, df)
    else:
        response = "❌ *Error:* No database, CSV, or MongoDB collection selected. Please connect to a data source."

    # Append AI response to session state
    st.session_state.messages.append({"role": "AI", "content": response})
    logger.info(f"AI response appended to session state: {response}")

# Handle user input from text input
def handle_user_input():
    user_query = st.session_state.get("user_input", "").strip()
    if user_query:
        process_query(user_query)
        # Clear the input by setting it to empty string
        st.session_state.user_input = ""

# Function to record audio
def record_audio(duration=5, fs=44100):
    st.write(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return recording

# Function to save audio to a WAV file
def save_audio(recording, fs, filename="output.wav"):
    wavio.write(filename, recording, fs, sampwidth=2)
    return filename

# Function to transcribe audio to text with ambient noise adjustment and retry mechanism
def transcribe_audio(filename, retries=3):
    recognizer = sr.Recognizer()
    for attempt in range(retries):
        try:
            with sr.AudioFile(filename) as source:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            logger.info(f"Audio transcribed successfully: {text}")
            return text
        except sr.UnknownValueError:
            return "❌ Could not understand the audio."
        except sr.RequestError:
            if attempt < retries - 1:
                continue  # Retry on request error
            return "❌ Could not connect to the service."
        except Exception as e:
            logger.error(f"Audio Transcription Error: {e}")
            return f"❌ An unexpected error occurred: {e}"

# Handle voice input and transcription
def handle_voice_input():
    with st.spinner("🎤 Recording..."):
        recording = record_audio(duration=5)  # Using existing record_audio function
        filename = save_audio(recording, 44100)  # Save the recorded file
        transcription = transcribe_audio(filename)
    if transcription and not transcription.startswith("❌"):
        st.write(f"🎤 You said: {transcription}")
        process_query(transcription)  # Pass transcribed text for further processing
    else:
        st.error(transcription)

# Initialize chat interface
st.title("💬 Chat with MySQL/CSV/MongoDB")

# Sidebar: Data source selection and connection form
with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("This is a simple chat application using MySQL, CSV, or MongoDB. Connect to a database or upload a CSV file.")

    st.markdown("---")

    # Data Source Selection
    data_source = st.radio(
        "Select Data Source",
        ("None", "MySQL", "MongoDB", "CSV"),
        index=0,
        key="data_source_radio"
    )

    # Function to handle data source selection
    def select_data_source():
        selected = st.session_state.data_source_radio
        if selected == "None":
            # Disconnect all data sources
            st.session_state.query_db = None
            st.session_state.schema_db = None
            st.session_state.mongo_db = None
            st.session_state.mongo_collection = None
            st.session_state.csv_data = None
            reset_chat_history()
        elif selected == "MySQL":
            # MySQL connection settings
            host = st.text_input("Host", value=DB_HOST, key="Host")
            port = st.text_input("Port", value=DB_PORT, key="Port")
            user = st.text_input("User", value=DB_USER, key="User")
            password = st.text_input("Password", type="password", value=DB_PASSWORD, key="Password")
            database = st.text_input("Database", value=DB_NAME, key="Database")

            if st.button("Connect to MySQL", key="connect_mysql"):
                with st.spinner("🔗 Connecting to MySQL..."):
                    try:
                        query_db, schema_db = init_databases(user, password, host, port, database)
                        st.session_state.query_db = query_db
                        st.session_state.schema_db = schema_db
                        # Deactivate other data sources
                        st.session_state.mongo_db = None
                        st.session_state.mongo_collection = None
                        st.session_state.csv_data = None
                        reset_chat_history()  # Reset chat history
                        st.success("✅ Connected to MySQL!")
                    except Exception as e:
                        st.error(f"❌ Failed to connect to MySQL: {parse_error_message(e)}")

        elif selected == "MongoDB":
            # MongoDB connection settings
            mongo_db_uri = st.text_input("MongoDB URI", value=MONGO_DB_URI, key="mongo_uri")
            mongo_db_name = st.text_input("MongoDB Database Name", value=MONGO_DB_NAME, key="mongo_dbname")

            if st.button("Connect to MongoDB", key="connect_mongo"):
                with st.spinner("🔗 Connecting to MongoDB..."):
                    mongo_db = init_mongo_db(mongo_db_uri, mongo_db_name)
                    if mongo_db is not None:
                        st.session_state.mongo_db = mongo_db
                        # Deactivate other data sources
                        st.session_state.query_db = None
                        st.session_state.schema_db = None
                        st.session_state.csv_data = None
                        reset_chat_history()  # Reset chat history
                        st.success("✅ Connected to MongoDB!")
                    else:
                        st.error("❌ Failed to connect to MongoDB.")

            # Choose a collection from the connected MongoDB
            if "mongo_db" in st.session_state and st.session_state.mongo_db is not None:
                collection_name = st.text_input("MongoDB Collection", key="collection_name")
                if st.button("Select Collection", key="select_collection"):
                    collection = get_mongo_collection(st.session_state.mongo_db, collection_name)
                    if collection is not None:
                        st.session_state.mongo_collection = collection
                        reset_chat_history()  # Reset chat history
                        st.success(f"✅ Selected collection '{collection_name}'!")

        elif selected == "CSV":
            # CSV file upload
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
            if uploaded_file is not None:
                load_csv(uploaded_file)
                # Deactivate other data sources
                st.session_state.query_db = None
                st.session_state.schema_db = None
                st.session_state.mongo_db = None
                st.session_state.mongo_collection = None
                reset_chat_history()  # Reset chat history after uploading a new CSV
                st.success(f"✅ Uploaded CSV: {uploaded_file.name}")

            # Option to display CSV schema if file is uploaded
            if "csv_data" in st.session_state and st.session_state.csv_data is not None and not st.session_state.csv_data.empty:
                st.button("📊 Display CSV Schema", on_click=display_csv_schema, key="display_schema_button")

    # Call the data source selection handler
    select_data_source()

    st.markdown("---")

    # Display active data source
    st.write("### Active Data Source")
    if st.session_state.get("query_db") is not None:
        st.success("✅ MySQL is connected.")
    elif st.session_state.get("mongo_db") is not None:
        st.success("✅ MongoDB is connected.")
    elif st.session_state.get("csv_data") is not None and not st.session_state.csv_data.empty:
        st.success("✅ CSV file is uploaded.")
    else:
        st.info("🔌 No data source connected.")

# Create a container for the input components
input_container = st.container()

with input_container:
    # Text input for user query with on_change callback
    user_query = st.text_input("Type a message and press Enter...", key="user_input", on_change=handle_user_input)

    # Create a column for the Record Voice button
    button_col = st.columns([1])

    with button_col[0]:
        record_button = st.button("🎤 Record Voice", key="voice_button")

    # Handle Record Voice button click
    if record_button:
        handle_voice_input()

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
