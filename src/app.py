from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import pymysql  # Ensure pymysql is installed
import sqlalchemy
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
import logging
import re
from datetime import datetime
import os
import pandas as pd  # For displaying query results
import graphviz  # For creating ER diagrams
import matplotlib.pyplot as plt
import plotly.express as px

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3307")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "kedar")
DB_NAME = os.getenv("DB_NAME", "friends")

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="💬 Chat with MySQL",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the ChatGroq model
chat_groq = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, temperature=0)

# Initialize the SQL Database connections
def init_databases(user: str, password: str, host: str, port: str, database: str) -> tuple[SQLDatabase, SQLDatabase]:
    try:
        db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        logger.info(f"Database URI for Queries: {db_uri}")
        query_db = SQLDatabase.from_uri(db_uri)
        schema_db = SQLDatabase.from_uri(db_uri)
        logger.info("Initialized separate database connections for queries and schema.")
        return query_db, schema_db
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise e

# Sanitize SQL queries by removing unwanted backslashes before underscores
def sanitize_query(query: str) -> str:
    return query.replace('\\', '')

# Function to parse and generate user-friendly error messages
def parse_error_message(exception: Exception) -> str:
    if isinstance(exception, pymysql.err.OperationalError):
        error_code = exception.args[0]
        error_msg = exception.args[1]
        if error_code == 1054:
            match = re.search(r"Unknown column '(.+?)' in 'field list'", error_msg)
            if match:
                column = match.group(1)
                return f"❌ **Error:** The column '{column}' does not exist in the specified table. Please check your query and try again."
            return "❌ **Error:** There was an issue with your query. Please verify the column names and try again."
        elif error_code == 1146:
            match = re.search(r"Table '(.+?)' doesn't exist", error_msg)
            if match:
                table = match.group(1)
                return f"❌ **Error:** The table '{table}' does not exist in the database. Please check your query and try again."
            return "❌ **Error:** There was an issue with your query. Please verify the table names and try again."
        else:
            return "❌ **Error:** There was an issue executing your query. Please ensure it's correct and try again."
    elif isinstance(exception, pymysql.err.ProgrammingError):
        return "❌ **Error:** There was a syntax error in your query. Please review it and try again."
    else:
        return "❌ **Error:** An unexpected error occurred while processing your request. Please try again later."

# Get SQL chain for the model interaction (Queries)
def get_sql_chain(query_db: SQLDatabase, schema_db: SQLDatabase):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History:
    {chat_history}

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
                    create_stmt = schema_db.run(f"SHOW CREATE TABLE {table};")
                    schema_info += create_stmt + "\n\n"
            logger.info("Successfully fetched schema information.")
            return schema_info
        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            return ""

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.invoke
        | StrOutputParser()
    )

# Generate the SQL response based on the user query and database state
def get_response(user_query: str, query_db: SQLDatabase, schema_db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(query_db, schema_db)

    try:
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, the question, the SQL query, and the SQL response, write a natural language response.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

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
            "chat_history": chat_history,
        })
        logger.info(f"Final Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        user_friendly_message = parse_error_message(e)
        return user_friendly_message

# Function to visualize data from a selected table
def visualize_table_data(engine: Engine, table_name: str):
    try:
        with engine.connect() as connection:
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 100", connection)
        
        st.subheader(f"Data Preview: {table_name}")
        st.dataframe(df)

        # User can choose which field to visualize
        st.subheader("Visualize Data")
        numeric_columns = df.select_dtypes(include=["float", "int"]).columns.tolist()
        non_numeric_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Display options for numeric columns (e.g., bar chart, line chart)
        if numeric_columns:
            st.subheader("Numeric Data Visualization")
            x_axis = st.selectbox("Select X-axis for Numeric Chart", numeric_columns, key="x_axis_num")
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart"], key="chart_type")

            if chart_type == "Bar Chart":
                fig, ax = plt.subplots()
                df[x_axis].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart of {x_axis}")
                st.pyplot(fig)
            elif chart_type == "Line Chart":
                fig = px.line(df, x=df.index, y=x_axis, title=f"Line Chart of {x_axis}")
                st.plotly_chart(fig)

        # Display options for non-numeric columns (e.g., pie chart, bar chart)
        if non_numeric_columns:
            st.subheader("Categorical Data Visualization")
            x_axis_cat = st.selectbox("Select X-axis for Categorical Chart", non_numeric_columns, key="x_axis_cat")
            chart_type_cat = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"], key="chart_type_cat")

            if chart_type_cat == "Bar Chart":
                fig, ax = plt.subplots()
                df[x_axis_cat].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart of {x_axis_cat}")
                st.pyplot(fig)
            elif chart_type_cat == "Pie Chart":
                fig = px.pie(df, names=x_axis_cat, title=f"Pie Chart of {x_axis_cat}")
                st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error: Could not retrieve data from {table_name}. {e}")

# Function to generate improved ER diagram with table and column selection
def generate_improved_er_diagram(schema_info: dict, fk_info: list):
    try:
        # Allow users to select which tables and columns to visualize
        selected_tables = st.multiselect("Select Tables to Include in ER Diagram", list(schema_info.keys()), default=list(schema_info.keys()))

        if not selected_tables:
            st.warning("⚠️ Please select at least one table to visualize.")
            return

        dot = graphviz.Digraph(comment='Enhanced ER Diagram', format='png')
        
        # Add nodes for each selected table
        for table in selected_tables:
            columns = schema_info[table]
            fields = "\\l".join(columns) + "\\l"  # Left-justified labels
            dot.node(table, f"{table}|{{{fields}}}", shape='record')
        
        # Add edges based on foreign keys for selected tables
        for fk in fk_info:
            if fk["table"] in selected_tables and fk["ref_table"] in selected_tables:
                dot.edge(fk["table"], fk["ref_table"], label=f"{fk['column']} → {fk['ref_column']}", arrowhead='vee')
        
        st.graphviz_chart(dot.source)
        st.success("✅ ER diagram generated successfully.")
    except Exception as e:
        st.error(f"❌ Error: Failed to generate ER diagram. {e}")

# Improved database schema visualization function
def visualize_database(engine: Engine):
    schema_info = fetch_schema_details(engine)
    if not schema_info:
        st.error("❌ Error: Unable to fetch schema details. Please ensure your database has tables and you have the necessary permissions.")
        return

    fk_info = fetch_foreign_keys(engine)
    if not fk_info:
        st.warning("⚠️ No foreign key relationships found. The ER diagram will only display tables without relationships.")

    # Generate improved ER diagram
    generate_improved_er_diagram(schema_info, fk_info)

    # Option to visualize data from specific tables
    st.subheader("📊 Table Data Visualization")
    table_name = st.selectbox("Select Table for Data Visualization", list(schema_info.keys()))
    if table_name:
        visualize_table_data(engine, table_name)

# Function to fetch schema details using SQLAlchemy Inspector
def fetch_schema_details(engine: Engine):
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            st.warning("⚠️ No tables found in the database.")
            return {}

        schema_info = {}
        for table in tables:
            columns = inspector.get_columns(table)
            column_names = [col['name'] for col in columns]
            schema_info[table] = column_names

        st.write("**Schema Information:**")
        st.json(schema_info)
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching schema details: {e}")
        st.error(f"❌ Error: Unable to fetch schema details.")
        return {}

# Function to fetch foreign key relationships
def fetch_foreign_keys(engine: Engine):
    try:
        inspector = inspect(engine)
        fk_info = []

        for table in inspector.get_table_names():
            foreign_keys = inspector.get_foreign_keys(table)
            for fk in foreign_keys:
                fk_info.append({
                    'table': table,
                    'column': fk['constrained_columns'][0],
                    'ref_table': fk['referred_table'],
                    'ref_column': fk['referred_columns'][0]
                })

        if fk_info:
            st.write("**Foreign Key Information:**")
            st.json(fk_info)
        else:
            st.warning("⚠️ No foreign key relationships found.")
        return fk_info
    except Exception as e:
        logger.error(f"Error fetching foreign keys: {e}")
        st.error(f"❌ Error: Unable to fetch foreign key information.")
        return []

# Streamlit configuration for UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

st.title("💬 Chat with MySQL")


# Database connection form
with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    host = st.text_input("Host", value=DB_HOST, key="Host")
    port = st.text_input("Port", value=DB_PORT, key="Port")
    user = st.text_input("User", value=DB_USER, key="User")
    password = st.text_input("Password", type="password", value=DB_PASSWORD, key="Password")
    database = st.text_input("Database", value=DB_NAME, key="Database")

    if st.button("Connect"):
        with st.spinner("🔗 Connecting to database..."):
            try:
                query_db, schema_db = init_databases(user, password, host, port, database)
                st.session_state.query_db = query_db
                st.session_state.schema_db = schema_db
                st.success("✅ Connected to database!")
            except Exception as e:
                st.error(f"❌ Failed to connect to database: {parse_error_message(e)}")

# Data Visualization Button
if st.sidebar.button("📊 Data Visualization"):
    if "schema_db" in st.session_state and "query_db" in st.session_state:
        with st.spinner("🔍 Fetching schema and generating diagram..."):
            try:
                engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
                visualize_database(engine)
            except Exception as e:
                st.error(f"❌ Failed to visualize database: {parse_error_message(e)}")
    else:
        st.error("❌ Database not connected. Please connect to the database first.")

# Display the chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input and responses
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if "query_db" in st.session_state and "schema_db" in st.session_state:
            response = get_response(user_query, st.session_state.query_db, st.session_state.schema_db, st.session_state.chat_history)
        else:
            response = "❌ **Error:** Database not connected. Please connect to the database first."
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
