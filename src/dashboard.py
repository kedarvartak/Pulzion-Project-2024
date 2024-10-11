import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import json

# Configuration
TABLE_NAME = "sales_tb"  # Update this if your table name is different

# Connect to MySQL using SQLAlchemy
def create_engine_connection():
    try:
        engine = create_engine(
            "mysql+pymysql://root:kedar@localhost:3307/retail_sales_db"
        )
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        return None

# Fetch data from the database based on query
def fetch_data(query):
    try:
        engine = create_engine_connection()
        if engine is None:
            print("Failed to create engine.")
            return pd.DataFrame()
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as err:
        print(f"Error fetching data: {err}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Retrieve the latest query from file
def get_query_from_file():
    try:
        with open('latest_query.json', 'r') as file:
            data = json.load(file)
        return data.get('query', None)
    except FileNotFoundError:
        print("File not found. Ensure that the main application has created the file.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON. Ensure that the file is formatted correctly.")
        return None

# Suggested visualization based on data analysis
def suggest_visualization(df):
    possible_visualizations = []
    
    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Suggest visualizations based on data types
    if categorical_columns and numerical_columns:
        possible_visualizations.append('bar')
    time_related = ['Date', 'Year', 'Month', 'Day']
    has_time = any(col in df.columns for col in time_related)
    if has_time and numerical_columns:
        possible_visualizations.append('line')
    if len(numerical_columns) >= 2:
        possible_visualizations.append('scatter')
    if 'Gender' in categorical_columns and 'TotalSales' in df.columns:
        possible_visualizations.append('pie')
    if numerical_columns:
        possible_visualizations.append('box')
    if 'Age' in df.columns or numerical_columns:
        possible_visualizations.append('histogram')

    # Order preference for visualization
    preference_order = ['bar', 'line', 'scatter', 'pie', 'box', 'histogram']
    suggested = next((chart for chart in preference_order if chart in possible_visualizations), None)
    return suggested, possible_visualizations

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Sales Dashboard"

# Fetch query from file and set it as the default query
query = get_query_from_file()
if query:
    df = fetch_data(query)
else:
    print("No query found. Using default query.")
    df = pd.DataFrame()  # Empty DataFrame as fallback

# Layout of the app
app.layout = html.Div([
    html.H1("📊 Sales Dashboard"),
    
    # Display loaded query for user context
    html.Label(f"Loaded Query: {query if query else 'No query found'}"),
    
    # Suggested visualization output
    html.Div(id='suggested-visualization', style={'font-weight': 'bold', 'margin-top': '10px'}),
    
    # Visualization selection based on data
    html.Label("Choose Visualization Type:"),
    dcc.Dropdown(
        id='visualization-dropdown',
        style={'width': '80%', 'margin-bottom': '20px'}
    ),
    
    # Display the chosen visualization
    html.Div(id="visualization"),
    
], style={'padding': '20px', 'maxWidth': '900px', 'margin': 'auto'})

# Callback to update the suggested visualization based on the fetched data
@app.callback(
    [Output('suggested-visualization', 'children'),
     Output('visualization-dropdown', 'options'),
     Output('visualization-dropdown', 'value')],
    [Input('visualization-dropdown', 'value')]
)
def update_suggestion(_):
    # Use the loaded query's data
    suggested, possible_visualizations = suggest_visualization(df)
    
    # Set dropdown options
    options = [{'label': viz.capitalize(), 'value': viz} for viz in possible_visualizations]
    default_value = suggested if suggested in possible_visualizations else (options[0]['value'] if options else None)

    return f"Suggested Visualization: {suggested.capitalize()}" if suggested else "No suggestion available", options, default_value

# Callback to render the graph based on the selected visualization type
@app.callback(
    Output('visualization', 'children'),
    [Input('visualization-dropdown', 'value')]
)
def update_graph(selected_chart):
    if not selected_chart:
        return "No visualization selected."
    
    if df.empty:
        return "❌ No data to display. Check your query or data source."

    # Define x and y for the plot
    x_col = df.columns[0]
    y_col = df.columns[1] if len(df.columns) > 1 else None

    # Render appropriate plot
    if selected_chart == 'bar' and y_col:
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template='simple_white')
    elif selected_chart == 'line' and y_col:
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", template='simple_white')
    elif selected_chart == 'scatter' and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", template='simple_white')
    elif selected_chart == 'pie' and y_col:
        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} by {x_col}", template='simple_white')
    elif selected_chart == 'box' and y_col:
        fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template='simple_white')
    elif selected_chart == 'histogram':
        fig = px.histogram(df, x=x_col, nbins=20, title=f"{x_col} Distribution", template='simple_white')
    else:
        return "❌ Unable to generate visualization with the provided data and selection."

    # Style and return the plot
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,  
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return dcc.Graph(figure=fig)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
