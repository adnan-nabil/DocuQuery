from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.database_create import client, embeddings, collection_name

#state definition
class mystate(TypedDict):
    question: str
    answer: str
    context: str
    history: Annotated[List, add_messages]

# Retrieval Node
def ret(state: mystate):

    query = state["question"]
    query_vector = embeddings.embed_query(query)
    
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
        with_payload=True
    )

    context = []
    for res in response.points:
        content = res.payload.get('page_content', "")
        
        score = res.score
        metadata = res.payload.get('metadata', {})
        page_no = metadata.get('page', 'unknown')
        formatted_chunk = f"{content}, score:{score:.3f}, page:{page_no}"
        context.append(formatted_chunk)

    context_txt = "\n\n".join(context)
    
    return {"context": context_txt}

# Generation Node
def gen(state: mystate):
    llm = ChatOllama(model="llama3.2", temperature=0)

    current_history = state.get("history", [])
    memory = state.get("history", [])[-8:]
    print(f"\n--- DEBUG: Total messages in history: {len(current_history)} ---")

    myprompt = f"""You are a strict chat assistant. 
    Analyze the provided context and answer the user question ONLY using the provided context or your conversation history.
    
    STRICT FORMAT RULES:
    1. Answer ONLY based on the context.
    2. MUST provide  Confidence score: [Score] and Page: [Page Number] for every new response"
    3. If not in context, reply EXACTLY: "This information is not present in the provided document."
    4. Do not invent any facts.
    5. if the user asks about a previous answer or a citation for it, use your conversation history to provide the source.
    6. you can refer to the conversation history to understand references like 'it', 'last question', or 'that policy'

    Context:
    {state['context']}"""

    input_message = [SystemMessage(content=myprompt)] + memory + [HumanMessage(content=state["question"])] 

    response = llm.invoke(input_message)
    
    
    return {
        "answer": response.content,
        "history": [
            HumanMessage(content=state["question"]), 
            AIMessage(content=response.content)
        ]
    }


#Connect Nodes Internally
pipe = StateGraph(mystate)

pipe.add_node("ret_node", ret)
pipe.add_node("gen_node", gen)

pipe.add_edge("ret_node", "gen_node")
pipe.add_edge("gen_node", END)

pipe.set_entry_point("ret_node")


memory_saver = MemorySaver()
sys = pipe.compile(checkpointer=memory_saver)



