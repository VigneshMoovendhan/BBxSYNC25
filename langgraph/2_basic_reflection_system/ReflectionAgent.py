# ======================================================
# 🐦 Twitter Post Generator with Reflection (LangGraph)
# ======================================================

from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
import os

# =========================
# 🔧 1. Setup Environment
# =========================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found in .env file.")

# =========================
# 🧩 2. Define Prompts
# =========================
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Twitter tech influencer assistant tasked with writing viral tech tweets. "
            "Generate the best possible tweet for the user's topic or request. "
            "If critique is given, improve the previous tweet accordingly.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer reviewing a tweet. "
            "Provide constructive critique, specific suggestions on tone, clarity, length, and virality potential.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# =========================
# 🤖 3. Initialize Models & Chains
# =========================
llm = ChatOpenAI(model="gpt-4o")
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# =========================
# 🔁 4. Define Graph Nodes
# =========================
REFLECT = "reflect"
GENERATE = "generate"

graph = MessageGraph()

def generate_node(state: List[BaseMessage]) -> BaseMessage:
    """Generates a tweet based on the input conversation."""
    print("\n🧠 Generating tweet based on conversation...")
    result = generation_chain.invoke({"messages": state})
    print(f"📝 Generated Tweet: {result.content}")
    return result

def reflect_node(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Reflects on the last tweet and gives feedback."""
    print("\n🔍 Reflecting on tweet for improvement...")
    response = reflection_chain.invoke({"messages": messages})
    print(f"💬 Reflection: {response.content}")
    return [HumanMessage(content=response.content)]

# =========================
# 🧠 5. Build the Graph
# =========================
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]) -> str:
    """Continue reflecting until a max of 4 messages."""
    if len(state) > 4:
        print("\n🏁 Ending reflection cycle (max depth reached).")
        return END
    print("\n🔁 Continuing to reflection stage...")
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# =========================
# 🗺️ 6. Visualize Graph
# =========================
print("\n🧱 ASCII Representation of the Graph:")
try:
    print(app.get_graph().print_ascii())
except AttributeError:
    print("⚠️ Visualization not supported in this version of LangGraph.")

try:
    print("\n🪶 Mermaid Diagram (for docs):")
    print(app.get_graph().draw_mermaid())
except Exception as e:
    print("⚠️ Could not generate Mermaid diagram:", e)

# =========================
# 🚀 7. Run Example
# =========================
print("\n🚀 Starting workflow: Generating & refining a tweet...\n")

try:
    response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))
    print("\n===============================")
    print("✅ FINAL OUTPUT MESSAGES:")
    print("===============================")
    for msg in response:
        role = msg.__class__.__name__
        print(f"{role}: {msg.content}\n")
except Exception as e:
    print(f"❌ Error during execution: {e}")
