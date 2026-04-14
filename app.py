import os
import streamlit as st
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
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="AI Fleet Commander", page_icon="🚛", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem;}
    .sub-header {font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;}
    .report-box {padding: 20px; border-radius: 10px; border: 1px solid #E5E7EB; background-color: #F9FAFB;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚛 AI Fleet Commander</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Agentic Predictive Maintenance & Autonomous Diagnostics</p>', unsafe_allow_html=True)

# Sidebar for Setup
with st.sidebar:
    st.header("⚙️ System Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Required to run the AI reasoning engine.")
    os.environ["GROQ_API_KEY"] = api_key
    st.markdown("---")
    st.markdown("**Agent Status:** " + ("🟢 Online" if api_key else "🔴 Offline (Needs Key)"))

# ==========================================
# 2. RAG KNOWLEDGE BASE
# ==========================================
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Initializes the FAISS vector database from the PDF manual."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    manual_path = "manuals/maintenance_guide.pdf"
    
    if os.path.exists(manual_path):
        loader = PyPDFLoader(manual_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return FAISS.from_documents(splitter.split_documents(docs), embeddings)
    else:
        # Fallback if PDF is missing
        return FAISS.from_texts(["Standard rules: Oil change every 5k miles. Ground vehicle if temp > 110C or vibration > 3.0."], embeddings)

vector_store = load_knowledge_base()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. STRUCTURED OUTPUT SCHEMA
# ==========================================
class FleetReport(BaseModel):
    health_summary: str = Field(description="Clear summary of vehicle status and risk level.")
    action_plan: str = Field(description="Step-by-step required maintenance actions.")
    sources: str = Field(description="Exact references from the manual used to make the decision.")
    disclaimer: str = Field(description="Safety warnings and operational disclaimers.")

# ==========================================
# 4. LANGGRAPH AGENT ENGINE
# ==========================================
class AgentState(TypedDict):
    telemetry: dict
    risk_detected: bool
    manual_excerpts: List[Document]
    final_report: dict

def evaluation_node(state: AgentState):
    """Agent evaluates if the telemetry crosses danger thresholds."""
    data = state["telemetry"]
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze telemetry. If Vibration > 3.0, Temp > 110, or Anomalies == 'Yes', reply 'CRITICAL'. Otherwise reply 'SAFE'."),
        ("user", "{data}")
    ])
    response = (prompt | llm).invoke({"data": data})
    return {"risk_detected": "CRITICAL" in response.content.upper()}

def routing_logic(state: AgentState):
    """Routes the workflow based on the evaluation."""
    return "search_manuals" if state["risk_detected"] else "generate_safe_report"

def retrieval_node(state: AgentState):
    """Agent searches the vector database for specific fixes."""
    query = f"Fixes for {state['telemetry']['Vehicle Type']} with Vibration {state['telemetry']['Vibration (G)']} and Temp {state['telemetry']['Engine Temp (C)']}"
    return {"manual_excerpts": retriever.invoke(query)}

def critical_report_node(state: AgentState):
    """Agent generates a full RAG report for a vehicle at risk."""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1).with_structured_output(FleetReport)
    docs = "\n".join([d.page_content for d in state["manual_excerpts"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead AI Fleet Mechanic. Draft a structured report using ONLY the provided manual excerpts to address the telemetry issues."),
        ("user", "Telemetry: {data}\n\nManual Excerpts: {docs}")
    ])
    result = (prompt | llm).invoke({"data": state["telemetry"], "docs": docs})
    return {"final_report": result.model_dump()}

def safe_report_node(state: AgentState):
    """Agent fast-tracks a safe report without doing RAG."""
    return {"final_report": {
        "health_summary": "✅ Vehicle is operating within optimal safety parameters.",
        "action_plan": "Continue normal dispatch operations. No immediate service required.",
        "sources": "Standard Operational Thresholds.",
        "disclaimer": "Routine visual inspections still apply before dispatch."
    }}

# Compile the Agent
graph = StateGraph(AgentState)
graph.add_node("evaluate", evaluation_node)
graph.add_node("search_manuals", retrieval_node)
graph.add_node("generate_critical", critical_report_node)
graph.add_node("generate_safe_report", safe_report_node)

graph.set_entry_point("evaluate")
graph.add_conditional_edges("evaluate", routing_logic, {"search_manuals": "search_manuals", "generate_safe_report": "generate_safe_report"})
graph.add_edge("search_manuals", "generate_critical")
graph.add_edge("generate_critical", END)
graph.add_edge("generate_safe_report", END)
agent = graph.compile()

# ==========================================
# 5. DYNAMIC UI DASHBOARD
# ==========================================
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("📥 Live Telemetry Input")
    st.markdown("Enter real-time data from the vehicle sensors.")
    
    with st.container(border=True):
        v_id = st.text_input("Vehicle ID", "TRK-092")
        v_type = st.selectbox("Vehicle Type", ["Heavy Truck", "Delivery Van", "Sprinter"])
        
        c1, c2 = st.columns(2)
        with c1:
            usage = st.number_input("Usage Hours", min_value=0, value=2500, step=100)
            vib = st.slider("Vibration (G)", 0.0, 10.0, 1.2, step=0.1)
        with c2:
            temp = st.number_input("Engine Temp (°C)", min_value=50, value=90, step=1)
            anom = st.radio("Anomalies Detected?", ["No", "Yes"])

        run_btn = st.button("🚀 Run Agentic Diagnostics", use_container_width=True, type="primary")

with col2:
    st.subheader("🤖 Autonomous Agent Output")
    
    if run_btn:
        if not api_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar to activate the AI Agent.")
        else:
            telemetry_data = {
                "Vehicle ID": v_id, "Vehicle Type": v_type, 
                "Usage Hours": usage, "Vibration (G)": vib, 
                "Engine Temp (C)": temp, "Anomalies": anom
            }
            
            # Watch the Agent Work!
            with st.status("Agentizing Fleet Data...", expanded=True) as status:
                st.write("📊 Evaluating sensor thresholds...")
                result = agent.invoke({"telemetry": telemetry_data})
                
                if result["risk_detected"]:
                    st.write("⚠️ Risk detected! Agent routing to manuals...")
                    st.write("📚 Retrieving FAISS knowledge base...")
                    st.write("✍️ Synthesizing maintenance protocol...")
                else:
                    st.write("✅ Telemetry looks good. Bypassing manual retrieval...")
                
                status.update(label="Diagnostic Complete!", state="complete", expanded=False)
            
            # Display Professional Output
            report = result["final_report"]
            
            # Use metrics for a dashboard feel
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Temp", f"{temp} °C", delta="High" if temp > 110 else "Normal", delta_color="inverse")
            m2.metric("Vibration", f"{vib} G", delta="Critical" if vib > 3.0 else "Normal", delta_color="inverse")
            m3.metric("System Risk", "HIGH" if result["risk_detected"] else "LOW", delta_color="off")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Structured Report Tabs
            t1, t2, t3 = st.tabs(["📋 Health & Action Plan", "📚 Reference Sources", "⚠️ Disclaimers"])
            
            with t1:
                if result["risk_detected"]:
                    st.error(f"**Health Summary:**\n{report['health_summary']}")
                    st.warning(f"**Action Plan:**\n{report['action_plan']}")
                else:
                    st.success(f"**Health Summary:**\n{report['health_summary']}")
                    st.info(f"**Action Plan:**\n{report['action_plan']}")
            
            with t2:
                st.markdown(f"*{report['sources']}*")
            
            with t3:
                st.caption(report['disclaimer'])
    else:
        st.info("👈 Enter telemetry data on the left and click 'Run Agentic Diagnostics' to generate a report.")