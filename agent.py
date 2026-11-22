from ddgs import DDGS
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Internet par live search karta hai."""
    return str(DDGS().text(query, max_results=3))

tools = [web_search]
llm = init_chat_model("google_genai:gemini-2.5-flash").bind_tools(tools)

def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()
while (q := input("You: ")) not in ("exit", "quit"):
    print("Agent:", app.invoke({"messages": [("user", q)]})["messages"][-1].text)
