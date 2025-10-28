import json
import os
import logging
from typing import TypedDict, List, Any
from dotenv import load_dotenv
 
# Preserve your original imports (these must be available in your environment)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
 
 
# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
 
 
# --- 2. Load Environment Variables ---
try:
    logger.debug("Loading environment variables from .env")
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
 
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        raise ValueError(
            "Missing API keys. Ensure OPENAI_API_KEY and TAVILY_API_KEY are set in the .env file."
        )
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise SystemExit(1)
 
 
# --- 3. Define the State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    topic: str
    report: str
 
 
# --- 4. Define Tools & Models ---
try:
    logger.debug("Initializing tools and models")
    tavily_tool = TavilySearchResults(max_results=4, description="A search engine tool")
    tools = [tavily_tool]
    tool_node = ToolNode(tools=tools)
 
    # Initialize Chat model
    model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    # If your ChatOpenAI supports binding tools, keep this; otherwise, you can call tools via ToolNode.
    try:
        research_model = model.bind_tools(tools)
    except Exception:
        # If bind_tools isn't available, fallback to model (ToolNode will handle tool invocation).
        research_model = model
 
    logger.info("Tools and models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing tools or models: {e}")
    raise SystemExit(1)
 
 
# --- 5. Define Agent Nodes ---
def researcher_node(state: AgentState) -> dict:
    """
    Expects state['topic'] and returns a dict with 'messages' (response from model).
    """
    topic = state.get("topic", "")
    logger.info(f"Running Researcher Node for topic: {topic}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a world-class researcher. Use the search tool to find the most relevant and up-to-date information on the given topic."),
            ("human", "Please research the topic: {topic}"),
        ]
    )
    researcher_runnable = prompt | research_model
    try:
        response = researcher_runnable.invoke({"topic": topic})
        # response may be an AIMessage-like object, or a wrapper; put it into messages list.
        logger.debug(f"Researcher response (preview): {getattr(response, 'content', str(response))[:200]}...")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in researcher node: {e}")
        return {"messages": [AIMessage(content=f"Error in research: {str(e)}")]}
 
 
def analyst_node(state: AgentState) -> dict:
    """
    Expects the last message in state['messages'] to be a ToolMessage (search results).
    Returns a dict with 'messages' containing the analyst AI response.
    """
    logger.info("Running Analyst Node")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a senior research analyst. Synthesize the provided search results into a coherent list of key insights. Focus on the most important facts and figures. Start directly with the insights."),
            ("user", "Here are the search results: {search_results}. Please provide the key insights."),
        ]
    )
    analyst_runnable = prompt | model
 
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    if last_message is None:
        logger.error("Analyst node: no last message found in state")
        return {"messages": [AIMessage(content="Error: No search results provided to analyst.")]}
 
    # If last_message is ToolMessage, extract its content; otherwise try to use its content attribute/string.
    if isinstance(last_message, ToolMessage):
        search_content = last_message.content
    else:
        # Accept other message types gracefully
        search_content = getattr(last_message, "content", str(last_message))
 
    logger.debug(f"Analyst processing search results (preview): {str(search_content)[:200]}...")
    try:
        response = analyst_runnable.invoke({"search_results": search_content})
        logger.debug(f"Analyst insights (preview): {getattr(response, 'content', str(response))[:200]}...")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in analyst node: {e}")
        return {"messages": [AIMessage(content=f"Error in analysis: {str(e)}")]}
 
 
def writer_node(state: AgentState) -> dict:
    """
    Expects last message to be insights (AI or text). Returns a dict with 'messages' and 'report'.
    """
    logger.info("Running Writer Node")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional report writer. Format the provided insights into a clean, well-structured report with a title, an executive summary, and a section for key insights."),
            ("user", "Here are the key insights: {insights}. Please write the final report."),
        ]
    )
    writer_runnable = prompt | model
 
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    if last_message is None:
        logger.error("Writer node: no last message found in state")
        return {"messages": [AIMessage(content="Error: No insights provided to writer.")], "report": ""}
 
    insights = getattr(last_message, "content", str(last_message))
    logger.debug(f"Writer processing insights (preview): {insights[:200]}...")
    try:
        response = writer_runnable.invoke({"insights": insights})
        report_text = getattr(response, "content", str(response))
        logger.debug(f"Writer generated report (preview): {report_text[:200]}...")
        return {"messages": [response], "report": report_text}
    except Exception as e:
        logger.error(f"Error in writer node: {e}")
        return {"messages": [AIMessage(content=f"Error in report writing: {str(e)}")], "report": ""}
 
 
# --- 6. Build the Graph ---
try:
    logger.info("Building the agent graph")
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tool_node", tool_node)  # ToolNode handles invoking the configured tools
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
 
    # Entry point and edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "tool_node")
    workflow.add_edge("tool_node", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)
 
    app = workflow.compile()
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Error building graph: {e}")
    raise SystemExit(1)
 
 
# --- 7. Run the Graph ---
if __name__ == "__main__":
    logger.info("Starting New Research Run")
    research_topic = "Latest advancements in AI-powered drug discovery"
    inputs: AgentState = {
        "topic": research_topic,
        "messages": [HumanMessage(content=f"Research the topic: {research_topic}")],
        "report": "",
    }
 
    try:
        # stream outputs if supported by your StateGraph implementation
        try:
            for output in app.stream(inputs, {"recursion_limit": 10}):
                node_that_ran = list(output.keys())[0]
                logger.info(f"Output from node: {node_that_ran}")
                for msg in output[node_that_ran].get("messages", []):
                    if isinstance(msg, AIMessage):
                        if getattr(msg, "tool_calls", None):
                            logger.info(f"[AIMessage] Calls tool: {msg.tool_calls}")
                        else:
                            logger.info(f"[AIMessage] {getattr(msg, 'content', str(msg))[:200]}...")
                    elif isinstance(msg, ToolMessage):
                        logger.info(f"[ToolMessage] Result: {getattr(msg, 'content', str(msg))[:200]}...")
                    else:
                        logger.info(f"[Message] {str(msg)[:200]}...")
        except AttributeError:
            # If .stream isn't available, ignore streaming and do a single invoke below
            logger.debug(".stream not available on app; falling back to invoke()")
 
        final_state = app.invoke(inputs, {"recursion_limit": 10})
        final_report = final_state.get("report", "")
        logger.info("---=--- FINAL REPORT PREVIEW ---=---")
        logger.info(final_report[:1000] if final_report else "No report generated.")
        print(final_report)
 
        # Save the report to a file if present
        if final_report:
            try:
                with open("report.md", "w", encoding="utf-8") as f:
                    f.write(final_report)
                logger.info("Final report saved to report.md")
            except Exception as e:
                logger.error(f"Error saving report to file: {e}")
    except Exception as e:
        logger.error(f"Error running graph: {e}")
        raise SystemExit(1)
 