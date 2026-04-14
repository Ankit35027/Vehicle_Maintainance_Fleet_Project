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

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem;}
    .sub-header {font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚛 AI Fleet Commander</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Agentic Predictive Maintenance & Autonomous Diagnostics</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Configuration")
    api_key = st.text_input("Groq API Key", type="password")
    os.environ["GROQ_API_KEY"] = api_key
    st.markdown("---")
    st.markdown("**Agent Status:** " + ("🟢 Online" if api_key else "🔴 Offline (Needs Key)"))

# ==========================================
# 2. RAG KNOWLEDGE BASE
# ==========================================
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    manual_path = "manuals/maintenance_guide.pdf"
    if os.path.exists(manual_path):
        loader = PyPDFLoader(manual_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return FAISS.from_documents(splitter.split_documents(docs), embeddings)
    else:
        return FAISS.from_texts(["Standard rules: Replace brakes if poor. Ground vehicle if temp > 110C, vibration > 3.0, or battery < 12V. Check tire pressure below 30 PSI."], embeddings)

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
    """Agent evaluates an expanded set of telemetry factors."""
    data = state["telemetry"]
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    # 🟢 EXPANDED AI RULES: The LLM now looks at ALL the new factors
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the vehicle telemetry. Mark 'CRITICAL' if ANY of these conditions are met:
        - Vibration (G) > 3.0
        - Engine Temp (°C) > 110
        - Anomalies Detected == 'Yes'
        - Tire Pressure (PSI) < 30
        - Battery Voltage (V) < 12.0
        - Oil Quality (%) < 15
        - Brake Condition == 'Poor'
        Otherwise, reply exactly with 'SAFE'."""),
        ("user", "{data}")
    ])
    response = (prompt | llm).invoke({"data": data})
    return {"risk_detected": "CRITICAL" in response.content.upper()}

def routing_logic(state: AgentState):
    return "search_manuals" if state["risk_detected"] else "generate_safe_report"

def retrieval_node(state: AgentState):
    t = state['telemetry']
    query = f"Fixes for {t['Vehicle Type']} with Vibration {t['Vibration (G)']}, Temp {t['Engine Temp (C)']}, Battery {t['Battery Voltage (V)']}, Brakes {t['Brake Condition']}"
    return {"manual_excerpts": retriever.invoke(query)}

def critical_report_node(state: AgentState):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1).with_structured_output(FleetReport)
    docs = "\n".join([d.page_content for d in state["manual_excerpts"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead AI Fleet Mechanic. Draft a structured report using ONLY the provided manual excerpts to address the specific telemetry faults."),
        ("user", "Telemetry: {data}\n\nManual Excerpts: {docs}")
    ])
    result = (prompt | llm).invoke({"data": state["telemetry"], "docs": docs})
    return {"final_report": result.model_dump()}

def safe_report_node(state: AgentState):
    return {"final_report": {
        "health_summary": "✅ Vehicle is operating within optimal safety parameters across all major systems.",
        "action_plan": "Continue normal dispatch operations. No immediate service required.",
        "sources": "Standard Operational Thresholds.",
        "disclaimer": "Routine visual inspections still apply before dispatch."
    }}

# Compile Graph
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
# 5. DYNAMIC UI DASHBOARD (EXPANDED FACTORS)
# ==========================================
col1, col2 = st.columns([1.2, 1.5], gap="large")

with col1:
    st.subheader("📥 Comprehensive Telemetry")
    st.markdown("Adjust live sensors to simulate vehicle conditions.")
    
    with st.container(border=True):
        # Top Row: Basic Info
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        v_id = r1_c1.text_input("Vehicle ID", "TRK-092")
        v_type = r1_c2.selectbox("Type", ["Truck", "Van", "Car"])
        route = r1_c3.selectbox("Route", ["Highway", "Urban", "Rural"])
        
        st.markdown("---")
        
        # Second Row: Engine & Performance
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        usage = r2_c1.number_input("Usage Hours", value=2500, step=100)
        temp = r2_c2.number_input("Engine Temp (°C)", value=90, step=1)
        vib = r2_c3.slider("Vibration (G)", 0.0, 10.0, 1.2, step=0.1)
        
        # Third Row: Consumables & Systems
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        tire = r3_c1.slider("Tire Pressure (PSI)", 10, 60, 35)
        battery = r3_c2.slider("Battery (V)", 10.0, 15.0, 13.5, step=0.1)
        oil = r3_c3.slider("Oil Quality (%)", 0, 100, 85)
        
        # Fourth Row: Diagnostics
        r4_c1, r4_c2 = st.columns(2)
        brakes = r4_c1.selectbox("Brake Condition", ["Good", "Fair", "Poor"])
        anom = r4_c2.radio("System Anomalies?", ["No", "Yes"])

        run_btn = st.button("🚀 Run Agentic Diagnostics", use_container_width=True, type="primary")

with col2:
    st.subheader("🤖 Autonomous Agent Output")
    
    if run_btn:
        if not api_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        else:
            # Package ALL the new factors for the agent
            telemetry_data = {
                "Vehicle ID": v_id, "Vehicle Type": v_type, "Route": route,
                "Usage Hours": usage, "Engine Temp (C)": temp, "Vibration (G)": vib, 
                "Tire Pressure (PSI)": tire, "Battery Voltage (V)": battery, 
                "Oil Quality (%)": oil, "Brake Condition": brakes, "Anomalies": anom
            }
            
            with st.status("Agentizing Fleet Data...", expanded=True) as status:
                st.write("📊 Evaluating multi-sensor thresholds...")
                result = agent.invoke({"telemetry": telemetry_data})
                
                if result["risk_detected"]:
                    st.write("⚠️ Risk detected across sensor array! Routing to manuals...")
                    st.write("📚 Searching vector database for complex faults...")
                    st.write("✍️ Synthesizing maintenance protocol...")
                else:
                    st.write("✅ All systems nominal. Bypassing manual retrieval...")
                
                status.update(label="Diagnostic Complete!", state="complete", expanded=False)
            
            report = result["final_report"]
            
            # Dynamic Metrics Display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Engine Temp", f"{temp}°C", delta="High" if temp > 110 else "Normal", delta_color="inverse")
            m2.metric("Battery", f"{battery}V", delta="Low" if battery < 12.0 else "Good", delta_color="normal")
            m3.metric("Oil Quality", f"{oil}%", delta="Change Due" if oil < 15 else "Good", delta_color="normal")
            m4.metric("System Risk", "HIGH" if result["risk_detected"] else "LOW", delta_color="off")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
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
        st.info("👈 Adjust the telemetry sliders on the left and click 'Run Agentic Diagnostics'.")