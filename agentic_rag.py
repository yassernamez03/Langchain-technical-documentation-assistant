# import basics
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub

from supabase.client import Client, create_client
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model (local, free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (Groq) (temperature = 0)
# Note: previous model llama3-70b-8192 was decommissioned by Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# fetch a tool-calling prompt compatible with the agent builder
prompt = hub.pull("hwchase17/openai-functions-agent")

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "why is agentic rag better than naive rag?"})

# put the result on the screen
print(response["output"])