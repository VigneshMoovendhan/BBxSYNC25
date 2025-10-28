from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
import os

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# Define prompt templates for generation and reflection
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Twitter techie influencer assistant tasked with writing excellent Twitter posts. "
            "Generate the best Twitter post possible for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the LLM (OpenAI model)
llm = ChatOpenAI(model="gpt-4o")

# Define chains for generation and reflection
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# Define node names
REFLECT = "reflect"
GENERATE = "generate"

# Initialize the MessageGraph
graph = MessageGraph()

# Define node functions
def generate_node(state: List[BaseMessage]) -> BaseMessage:
    """
    Generate a tweet based on the input state (messages).
    """
    return generation_chain.invoke({"messages": state})

def reflect_node(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Critique the generated tweet and return a HumanMessage with feedback.
    """
    response = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=response.content)]

# Add nodes to the graph
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Set the entry point
graph.set_entry_point(GENERATE)

# Define conditional routing
def should_continue(state: List[BaseMessage]) -> str:
    """
    Determine whether to continue to reflection or end the workflow.
    Stops after 6 messages to prevent infinite loops.
    """
    if len(state) > 3:
        return END
    return REFLECT

# Add edges
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

# Compile the graph
app = graph.compile()

# Print graph visualizations
print("Mermaid Diagram:")
print(app.get_graph().draw_mermaid())
print("\nASCII Graph:")
app.get_graph().print_ascii()

# Run the workflow with an example input
try:
    response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))
    print("\nFinal Response:")
    for message in response:
        print(f"{message.__class__.__name__}: {message.content}")
except Exception as e:
    print(f"Error during execution: {e}")