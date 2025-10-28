# ======================================================
# 📝 AI Researcher + Revisor Workflow with Tool Execution
# ======================================================

import datetime
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langgraph.graph import MessageGraph, END
from langchain_community.tools import TavilySearchResults

# =========================
# 🔧 1. Load Environment
# =========================
load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

# =========================
# 🏗️ 2. Define Schemas
# =========================
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")

class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(description="Citations motivating your updated answer.")

# =========================
# 🧩 3. Setup Parsers & Tools
# =========================
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
json_parser = JsonOutputToolsParser(return_id=True)

tavily_tool = TavilySearchResults(max_results=5)

# =========================
# 📝 4. Actor Prompts & Chains
# =========================
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
first_responder_chain = first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_prompt = actor_prompt_template.partial(first_instruction="""
Revise your previous answer using the new information.
- Include numerical citations.
- Add a References section.
- Use previous critique to remove superfluous info.
- Max 250 words.
""")
revisor_chain = revise_prompt | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# =========================
# 🔧 5. Tool Execution Function
# =========================
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai: AIMessage = state[-1]
    if not hasattr(last_ai, "tool_calls") or not last_ai.tool_calls:
        return []

    tool_messages = []

    for call in last_ai.tool_calls:
        if call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = call["id"]
            search_queries = call["args"].get("search_queries", [])
            results = {q: tavily_tool.invoke(q) for q in search_queries}
            tool_messages.append(ToolMessage(content=json.dumps(results), tool_call_id=call_id))

    return tool_messages

# =========================
# 🔀 6. Build Message Graph
# =========================
graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(msg, ToolMessage) for msg in state)
    if count_tool_visits > MAX_ITERATIONS:
        print("🏁 Max iterations reached. Ending workflow.")
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

# =========================
# 🗺️ 7. Visualize Graph
# =========================
print("\n🧱 ASCII Graph:")
try:
    print(app.get_graph().print_ascii())
except AttributeError:
    print("⚠️ ASCII visualization not available.")

try:
    print("\n🪶 Mermaid Diagram:")
    print(app.get_graph().draw_mermaid())
except Exception as e:
    print("⚠️ Could not generate Mermaid diagram:", e)

# =========================
# 🚀 8. Run Example
# =========================
input_message = HumanMessage(content="Write about how small business can leverage AI to grow")
response = app.invoke(input_message)

print("\n===============================")
print("✅ Final Output:")
print("===============================")
for msg in response:
    role = msg.__class__.__name__
    if isinstance(msg, ToolMessage):
        content = json.loads(msg.content)
    else:
        content = getattr(msg, "content", "")
    print(f"{role}: {content}\n")
