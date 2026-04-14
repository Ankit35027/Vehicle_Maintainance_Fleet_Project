import os
import streamlit as st
import pandas as pd
from typing import TypedDict, List
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Agentic Fleet Manager", layout="wide")
st.title("🚛 Agentic AI Fleet Management Assistant")

# Sidebar for API Key
st.sidebar.markdown("### API Configuration")
os.environ["GROQ_API_KEY"] = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Get a free key at console.groq.com")

# ==========================================
# 2. RAG INITIALIZATION (FAISS)
# ==========================================
@st.cache_resource
def initialize_vector_store():
    """Loads PDF manuals into a FAISS vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    manual_path = "manuals/maintenance_guide.pdf"
    if not os.path.exists(manual_path):
        # Fallback empty vector store if no PDF is found yet
        return FAISS.from_texts(["Ensure regular oil changes and visual inspections. Check tire pressure weekly."], embeddings)
        
    loader = PyPDFLoader(manual_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(splits, embeddings)

vector_store = initialize_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. STRUCTURED OUTPUT DEFINITION
# ==========================================
class FleetRecommendation(BaseModel):
    health_summary: str = Field(description="Vehicle status & Risk assessment.")
    action_plan: str = Field(description="Recommended service & Timeline.")
    sources: str = Field(description="Maintenance manuals/refs used.")
    disclaimer: str = Field(description="Operational safety notices.")

# ==========================================
# 4. LANGGRAPH STATE & NODES
# ==========================================
class FleetAgentState(TypedDict):
    vehicle_data: dict          
    needs_maintenance: bool              
    retrieved_guidelines: List[Document] 
    final_recommendation: dict    

def reason_vehicle_health_node(state: FleetAgentState):
    """Agentic Step 1: Analyze telemetry to decide if deep RAG action is needed."""
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    
    # Analyze critical metrics to determine if the vehicle requires attention
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Fleet AI. Look at the vehicle telemetry. If Vibration_Levels > 3.0, Engine_Temperature > 110, or Anomalies_Detected == 1, reply ONLY with 'True'. Otherwise, reply ONLY with 'False'."),
        ("user", "Telemetry Data: {telemetry}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"telemetry": state["vehicle_data"]})
    
    needs_maintenance = "true" in response.content.lower()
    return {"needs_maintenance": needs_maintenance}

def route_next_step(state: FleetAgentState):
    """Routes the graph based on the AI's reasoning."""
    if state.get("needs_maintenance"):
        return "retrieve_guidelines"
    else:
        return "generate_healthy_report"

def retrieve_guidelines_node(state: FleetAgentState):
    """Agentic Step 2: Retrieves manuals based on vehicle condition."""
    v_type = state["vehicle_data"].get("Vehicle_Type", "Vehicle")
    vibration = state["vehicle_data"].get("Vibration_Levels", "Unknown")
    
    search_query = f"Maintenance guidelines for {v_type} with vibration level {vibration}."
    docs = retriever.invoke(search_query)
    return {"retrieved_guidelines": docs}

def generate_recommendation_node(state: FleetAgentState):
    """Agentic Step 3: Generates the structured fleet recommendation using RAG."""
    llm = ChatGroq(temperature=0.1, model_name="llama3-8b-8192")
    structured_llm = llm.with_structured_output(FleetRecommendation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI Fleet Manager. Analyze the telemetry data and apply the retrieved guidelines to create a structured maintenance plan."),
        ("user", "Telemetry Data: {telemetry}\n\nGuidelines from Manual: {guidelines}")
    ])
    
    chain = prompt | structured_llm 
    docs_text = "\n".join([d.page_content for d in state["retrieved_guidelines"]])
    
    response = chain.invoke({
        "telemetry": state["vehicle_data"],
        "guidelines": docs_text
    })
    
    return {"final_recommendation": response.model_dump()}

def generate_healthy_report_node(state: FleetAgentState):
    """Fast-Track Step: Generates a standard report if the vehicle is healthy."""
    rec = {
        "health_summary": "Vehicle is operating within normal parameters. No immediate risk detected based on current telemetry.",
        "action_plan": "Continue standard operations. No immediate maintenance scheduling required. Re-evaluate at next interval.",
        "sources": "Standard Operating Procedures (No specific emergency manual retrieved).",
        "disclaimer": "Always perform standard pre-trip visual inspections regardless of AI telemetry."
    }
    return {"final_recommendation": rec}

# ==========================================
# 5. COMPILE LANGGRAPH WORKFLOW
# ==========================================
workflow = StateGraph(FleetAgentState)

# Add all nodes
workflow.add_node("reason_vehicle_health", reason_vehicle_health_node)
workflow.add_node("retrieve_guidelines", retrieve_guidelines_node)
workflow.add_node("generate_recommendation", generate_recommendation_node)
workflow.add_node("generate_healthy_report", generate_healthy_report_node)

# Set Graph Entry Point
workflow.set_entry_point("reason_vehicle_health")

# Add Conditional Edge (The Agentic Router)
workflow.add_conditional_edges(
    "reason_vehicle_health",
    route_next_step,
    {
        "retrieve_guidelines": "retrieve_guidelines",
        "generate_healthy_report": "generate_healthy_report"
    }
)

# Standard Edges to End the Workflow
workflow.add_edge("retrieve_guidelines", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)
workflow.add_edge("generate_healthy_report", END)

# Compile
fleet_agent = workflow.compile()

# ==========================================
# 6. STREAMLIT UI INTEGRATION
# ==========================================
@st.cache_data
def load_data():
    dataset_name = 'fleet_maintenance_unbiased_40k.csv'
    if os.path.exists(dataset_name):
        return pd.read_csv(dataset_name)
    return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("Fleet Selection")
    selected_index = st.sidebar.number_input("Select Vehicle Row Index", min_value=0, max_value=len(df)-1, value=0, step=1)
    vehicle_data = df.iloc[selected_index].to_dict()
    
    st.subheader(f"📊 Current Telemetry: Vehicle #{selected_index}")
    st.json(vehicle_data)
    
    if st.button("Generate Autonomous Fleet Report", type="primary"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        else:
            with st.spinner("Agent is reasoning, retrieving manuals, and planning..."):
                try:
                    initial_state = {"vehicle_data": vehicle_data}
                    result = fleet_agent.invoke(initial_state)
                    rec = result["final_recommendation"]
                    
                    st.success("Analysis Complete!")
                    
                    # Layout identical to Rubric Requirements
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 📋 Health Summary")
                        st.info(rec["health_summary"])
                        st.markdown("### 📚 Sources")
                        st.write(rec["sources"])
                    with col2:
                        st.markdown("### 🛠️ Action Plan")
                        st.success(rec["action_plan"])
                        st.markdown("### ⚠️ Disclaimer")
                        st.warning(rec["disclaimer"])
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.error("Could not find 'fleet_maintenance_unbiased_40k.csv'. Please ensure it is in the same directory as app.py.")