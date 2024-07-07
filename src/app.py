import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from pyprojroot import here
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize the database and language model
db_path = str(here("data")) + "/sqldb.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the query chain
write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)
sql_execution_chain = write_query | execute_query

# Create the agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)

# Define the prompt template for DataFrame conversion
df_prompt_template = """
Given the following output:

{result}

1. Analyze the output and identify any necessary changes.
2. Make the required changes.
3. Format the output in a way that can be easily converted into a Pandas DataFrame.
4. Provide the data in JSON format suitable for conversion into a DataFrame, including column names if applicable.

Only provide the JSON data.
"""

dataframe_prompt = ChatPromptTemplate.from_template(df_prompt_template)
parser = JsonOutputParser()
dataframe_chain = dataframe_prompt | llm | parser

# Define session state
class SessionState:
    def __init__(self):
        self.user_question = ""
        self.plot_type = "Bar"
        self.df = None
        self.fig = None

# Initialize session state
session_state = SessionState()

# Streamlit application
st.title("Natural Language to SQL Query Application")

# Input from user
session_state.user_question = st.text_input("Enter your query:", session_state.user_question)
session_state.plot_type = st.selectbox("Select plot type:", ["Bar", "Line", "Pie"], index=["Bar", "Line", "Pie"].index(session_state.plot_type))

if session_state.user_question:
    sql_query_result = sql_execution_chain.invoke({"question": session_state.user_question})
    agent_result = agent_executor.invoke({"input": session_state.user_question})
    
    final_result = dataframe_chain.invoke({"result": sql_query_result})
    session_state.df = pd.DataFrame(final_result)

    # Plotting based on user choice
    if session_state.plot_type == "Bar":
        if len(session_state.df.columns) >= 2:
            session_state.fig = px.bar(session_state.df, x=session_state.df.columns[0], y=session_state.df.columns[1])
        else:
            st.write("Not enough data to plot a bar chart.")
    elif session_state.plot_type == "Line":
        if len(session_state.df.columns) >= 2:
            session_state.fig = px.line(session_state.df, x=session_state.df.columns[0], y=session_state.df.columns[1])
        else:
            st.write("Not enough data to plot a line chart.")
    elif session_state.plot_type == "Pie":
        if len(session_state.df.columns) >= 2:
            session_state.fig = px.pie(session_state.df, names=session_state.df.columns[0], values=session_state.df.columns[1])
        else:
            st.write("Not enough data to plot a pie chart.")
    
    st.write("SQL Query:")
    st.write(agent_result["output"])
    
    st.write("DataFrame:")
    st.write(session_state.df)
    
    if session_state.fig:
        st.plotly_chart(session_state.fig)
