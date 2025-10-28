# =========================
# 🧠 ReAct Agent Graph (LangGraph + LangChain)
# =========================

from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
import operator
from typing import Annotated, TypedDict, Union

# =========================
# 🔧 1. Environment Setup
# =========================
load_dotenv()

# =========================
# 🧩 2. Define Tools
# =========================
llm = ChatOpenAI(model="gpt-4")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic")
react_prompt = hub.pull("hwchase17/react")
tools = [get_system_time, search_tool]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)

# =========================
# 🧱 3. Define Agent State
# =========================
class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# =========================
# 🔁 4. Define Workflow Nodes
# =========================
def reason_node(state: AgentState):
    print("\n🧠 Reasoning Node Invoked...")
    agent_outcome = react_agent_runnable.invoke(state)
    print("🤖 Agent Outcome:", agent_outcome)
    return {"agent_outcome": agent_outcome}

def act_node(state: AgentState):
    print("\n⚙️ Action Node Invoked...")
    agent_action = state["agent_outcome"]

    tool_name = agent_action.tool
    tool_input = agent_action.tool_input
    print(f"🔍 Tool to execute: {tool_name} | Input: {tool_input}")

    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break

    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
        print(f"✅ Tool '{tool_name}' executed successfully. Output: {output}")
    else:
        output = f"⚠️ Tool '{tool_name}' not found"
        print(output)

    return {"intermediate_steps": [(agent_action, str(output))]}

# =========================
# 🔀 5. Define Graph Logic
# =========================
REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        print("\n🏁 Agent finished reasoning. Ending workflow.")
        return END
    else:
        print("\n➡️ Continuing to Action Node.")
        return ACT_NODE

graph = StateGraph(AgentState)
graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)
graph.add_node(ACT_NODE, act_node)
graph.add_conditional_edges(REASON_NODE, should_continue)
graph.add_edge(ACT_NODE, REASON_NODE)

# =========================
# 🚀 6. Compile and Run
# =========================
app = graph.compile()

print("\n🚀 Invoking agent graph...\n")
result = app.invoke({
    "input": "How many days ago was the latest SpaceX launch?",
    "agent_outcome": None,
    "intermediate_steps": []
})

print("\n===============================")
print("✅ FINAL RESULT:")
print(result["agent_outcome"].return_values["output"])
print("===============================")

# =========================
# 📊 7. Visualize Graph
# =========================
print("\n🧱 ASCII Representation of the Graph:")
try:
    print(app.get_graph().print_ascii())
except AttributeError:
    try:
        print(graph.print_ascii())
    except AttributeError:
        print("⚠️ Visualization not supported in this version of LangGraph.")

# Optional: Mermaid or PNG visualization (if Graphviz installed)
try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
    print("✅ Displayed workflow graph as image.")
except Exception as e:
    print("⚠️ Could not render image graph (try installing graphviz):", e)
