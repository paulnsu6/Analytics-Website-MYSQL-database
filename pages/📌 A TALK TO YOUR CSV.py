

import os
import pandas as pd
import streamlit as st
from pandasai import Agent
from PyPDF2 import PdfReader

# Title of the app
st.title("File Analysis Tool")

# Directory for preloaded files
data_dir = "dataV2"  # Directory where your preloaded files are stored
if not os.path.exists(data_dir):
    os.makedirs(data_dir)  # Create the directory if it doesn't exist

# Fetch all CSV and XLSX files from the directory
preloaded_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx'))]

# Allow user to upload their own file
uploaded_file = st.file_uploader("Upload a file for analysis (CSV, XLSX, PDF):", type=["csv", "xlsx", "pdf"])

# Combine preloaded files and uploaded file into a single selection
if uploaded_file:
    uploaded_filename = uploaded_file.name
    selected_file = uploaded_filename  # Track the uploaded file as the selected file
    st.write(f"You uploaded: **{uploaded_filename}**")
else:
    # If no file is uploaded, provide a dropdown to select from preloaded files
    if preloaded_files:
        selected_file = st.selectbox("Or select a preloaded file for analysis:", preloaded_files)
    else:
        st.warning("No preloaded files found in the 'data' directory. Please upload a file to proceed.")
        selected_file = None

# Process the selected or uploaded file
if selected_file:
    try:
        # Load data based on file type
        if uploaded_file:
            # If the file is uploaded
            if uploaded_filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_filename.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_filename.endswith('.pdf'):
                # Extract text from the PDF
                pdf_reader = PdfReader(uploaded_file)
                pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                df = pd.DataFrame({'PDF_Text': [pdf_text]})
        else:
            # If the file is preloaded
            file_path = os.path.join(data_dir, selected_file)
            if selected_file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif selected_file.endswith('.xlsx'):
                df = pd.read_excel(file_path)

        # Display the DataFrame
        st.write("Preview of the data:")
        st.dataframe(df)

        # Initialize the PandasAI Agent
        # agent = Agent(
        #     df,
        #     description="You are a data analysis agent. Your main goal is to help non-technical users to analyze data.",
        # )
        agent = Agent(
                df,
            description=(
                "You are a data analysis agent specializing in providing actionable insights. "
                "Your main goal is to help non-technical users analyze data and make informed decisions. "
                "Focus on understanding trends, patterns, and key factors from the data, and provide detailed, practical recommendations rather than just identifying specific values."
        ),
)
        # User query input
        user_query = st.text_input("Enter your analysis question:")

        if st.button("Run Analysis"):
            if user_query.strip():
                with st.spinner("Analyzing..."):
                    try:
                        response = agent.chat(user_query)
                        st.success("Analysis Complete!")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a question for analysis.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")


# import json
# import os
# from pandasai import Agent
# import streamlit as st
# import pandas as pd

# # By default, unless you choose a different LLM, it will use BambooLLM.
# # You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
# os.environ["PANDASAI_API_KEY"] = "$2a$10$CaKQ9kHgCVNpBfGlZHtbCemZRZi6F5hq2Y20r/Q6Yfcl/UrJp8O8e"




# # employees_df = pd.DataFrame('advanced_regression.csv')
# # salaries_df = pd.DataFrame('dataset.csv')


# # Load datasets
# advanced_regression = pd.read_csv('advanced_regression.csv')
# dataset = pd.read_csv('dataset.csv')




# # Pass the DataFrames instead of file paths
# agent = Agent(
#     # [advanced_regression],
#     'data.xlsx',
#     # ['dataset.csv','advanced_regression.csv'],
#     description="You are a data analysis agent. Your main goal is to help non-technical users to analyze data",
# )

# # Chat with the agent
# # response = agent.chat("provide a summary of the dataset ")
# # print(response)
# # st.write(response)

# response = agent.chat("summarise this data")
# if isinstance(response, dict):
#     response_json = json.dumps(response, indent=4)
#     print(response_json)
#     st.write(response_json)
# else:
#     print(response)
#     st.write(response)

























# from langchain import OpenAI
# from langchain.agents import create_pandas_dataframe_agent
# import pandas as pd
# from dotenv import load_dotenv 
# import json
# import streamlit as st

# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# def csv_tool(filename : str):

#     df = pd.read_csv(filename)
#     return create_pandas_dataframe_agent(lChatGoogleGenerativeAI(model="gemini-pro"), df, verbose=True)

# def ask_agent(agent, query):
#     """
#     Query an agent and return the response as a string.

#     Args:
#         agent: The agent to query.
#         query: The query to ask the agent.

#     Returns:
#         The response from the agent as a string.
#     """
#     # Prepare the prompt with query guidelines and formatting
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

#         3. If a line chart is more appropriate, your reply should look like this:
#            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

#         Note: We only accommodate two types of charts: "bar" and "line".

#         4. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         For example:
#            {"answer": "The Product with the highest Orders is '15143Exfo'"}

#         5. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )

#     # Run the prompt through the agent and capture the response.
#     response = agent.run(prompt)

#     # Return the response converted to a string.
#     return str(response)

# def decode_response(response: str) -> dict:
#     """This function converts the string response from the model to a dictionary object.

#     Args:
#         response (str): response from the model

#     Returns:
#         dict: dictionary with response data
#     """
#     return json.loads(response)

# def write_answer(response_dict: dict):
#     """
#     Write a response from an agent to a Streamlit app.

#     Args:
#         response_dict: The response from the agent.

#     Returns:
#         None.
#     """

#     # Check if the response is an answer.
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])

#     # Check if the response is a bar chart.
#     # Check if the response is a bar chart.
#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         try:
#             df_data = {
#                     col: [x[i] if isinstance(x, list) else x for x in data['data']]
#                     for i, col in enumerate(data['columns'])
#                 }       
#             df = pd.DataFrame(df_data)
#             df.set_index("Products", inplace=True)
#             st.bar_chart(df)
#         except ValueError:
#             print(f"Couldn't create DataFrame from data: {data}")

# # Check if the response is a line chart.
#     if "line" in response_dict:
#         data = response_dict["line"]
#         try:
#             df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
#             df = pd.DataFrame(df_data)
#             df.set_index("Products", inplace=True)
#             st.line_chart(df)
#         except ValueError:
#             print(f"Couldn't create DataFrame from data: {data}")


#     # Check if the response is a table.
#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)
# st.set_page_config(page_title="👨‍💻 Talk with your CSV")
# st.title("👨‍💻 Talk with your CSV")

# st.write("Please upload your CSV file below.")

# data = st.file_uploader("Upload a CSV" , type="csv")

# query = st.text_area("Send a Message")

# if st.button("Submit Query", type="primary"):
#     # Create an agent from the CSV file.
#     agent = csv_tool(data)

#     # Query the agent.
#     response = ask_agent(agent=agent, query=query)

#     # Decode the response.
#     decoded_response = decode_response(response)

#     # Write the response to the Streamlit app.
#     write_answer(decoded_response)