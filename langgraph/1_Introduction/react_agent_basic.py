from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")


# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.chat_models import ChatOpenAI
# import datetime

# load_dotenv()

# # -----------------------------
# # Step 1: Initialize OpenAI LLM
# # -----------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# # -----------------------------
# # Step 2: Setup tools
# # -----------------------------
# search_tool = TavilySearchResults(search_depth="basic")

# @tool
# def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
#     """ Returns the current date and time in the specified format """
#     current_time = datetime.datetime.now()
#     return current_time.strftime(format)

# tools = [search_tool, get_system_time]

# # -----------------------------
# # Step 3: Initialize agent
# # -----------------------------
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=True
# )

# # -----------------------------
# # Step 4: Run a query
# # -----------------------------
# response = agent.invoke(
#     "When was SpaceX's last launch and how many days ago was that from this instant?"
# )

# print(response)
