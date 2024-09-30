# Pulzion-2024-Web N App entry for team algoRIZZmers - Kedar Vartak, Tejas Kulkarni, Yash Kulkarni, Varad Kulkarni.

# 💬 Chat with MySQL/CSV/MongoDB

This project is a data-driven **chat interface** built with **Streamlit**, allowing users to query MySQL databases, CSV files, or MongoDB collections through natural language inputs. The application seamlessly integrates voice and text inputs to allow users to interact with various data sources in real time. The platform dynamically handles data switching between different databases, making it easy for users to access data from multiple sources in a user-friendly interface.

## 🌟 Features

### 1. **MySQL Querying**
   - **Connect to MySQL Database**: Users can input their MySQL database credentials to connect and query data.
   - **Automatic Query Generation**: The system generates SQL queries based on the user’s natural language questions.
   - **Data-Driven Responses**: Responses are generated by running queries against the connected database and returning meaningful results.

### 2. **CSV File Upload**
   - **Upload CSV Files**: Users can upload CSV files directly through the interface.
   - **CSV Querying**: The system processes user queries against the uploaded CSV file, retrieving data based on simple natural language questions.
   - **Dynamic Schema Display**: The system displays the schema (columns and types) of the uploaded CSV file, helping users understand the available data.

### 3. **MongoDB Integration**
   - **MongoDB Connection**: Users can connect to a MongoDB database by providing the necessary connection string and database name.
   - **MongoDB Querying**: Natural language queries are automatically translated into MongoDB queries, fetching relevant documents from the database.
   - **Collection Selection**: Users can choose specific MongoDB collections to run their queries against.

### 4. **Voice Input**
   - **Voice-to-Text Support**: Users can use voice commands to input queries. The system records and transcribes the voice input to text, which is then processed like any other query.
   - **Voice Recording**: Audio is captured using the **SoundDevice** library, allowing users to interact with the app using spoken language.

### 5. **Automatic Data Source Switching**
   - **Dynamic Data Source Management**: The platform automatically handles switching between MySQL, CSV, and MongoDB as users connect or disconnect from different sources.
   - **State Management**: Switching between databases resets chat history and ensures that each query is relevant to the selected data source.

### 6. **Natural Language Processing**
   - **LLM-Powered Query Translation**: The project leverages **Langchain** and **ChatGroq** models to translate user questions into SQL or MongoDB queries based on the connected data source schema.
   - **Error Handling**: Friendly error messages are generated in case of invalid queries or misconfigurations.

## 🛠️ Technologies Used

- **Streamlit**: A Python framework used to create the web interface and manage the chat interactions.
- **Langchain**: Powers the natural language processing to understand user queries and generate database queries (SQL/MongoDB).
- **ChatGroq**: A language model used to generate database-specific queries based on user inputs.
- **MySQL**: Handles relational data queries, allowing users to connect and query MySQL databases.
- **MongoDB**: NoSQL database integration for fetching documents based on natural language queries.
- **Pandas**: Used for handling CSV files and data manipulation.
- **SoundDevice**: Captures and processes audio input for voice commands.
- **SpeechRecognition**: Converts voice input into text for query processing.
- **pymysql**: Used to connect and execute SQL queries on MySQL databases.
- **pymongo**: Facilitates interaction with MongoDB for document retrieval.
- **dotenv**: Loads environment variables for secure configuration management.

## 🚀 How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/chat-with-mysql-csv-mongodb.git
   cd chat-with-mysql-csv-mongodb
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file with the following variables:
     ```
     GROQ_API_KEY=<Your ChatGroq API Key>
     DB_HOST=<MySQL Host>
     DB_PORT=<MySQL Port>
     DB_USER=<MySQL Username>
     DB_PASSWORD=<MySQL Password>
     DB_NAME=<MySQL Database Name>
     MONGO_DB_URI=<Your MongoDB Connection URI>
     MONGO_DB_NAME=<Your MongoDB Database Name>
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the Application**:
   - Open your browser and go to `http://localhost:8501` to start interacting with the app.

## 📝 Future Enhancements

- Support for more databases (e.g., PostgreSQL).
- Enhanced query parsing for more complex natural language queries.
- Improved UI/UX with better visualizations for data output.

