"""
CSV Agent (ReAct + REPL) - AI-Powered CSV Analysis with Chart Visualization
Auto-loads all Excel/CSV files from data/ folder
Implements: Context Creation → Prompt Augmentation → Code Generation → Execution → Chart Generation
Uses LangChain ReAct agent with PythonREPL for safe code execution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
import html as html_lib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# =============================================================================
# Round-Robin LLM Provider (Groq ↔ Gemini rotation)
# =============================================================================

# All keys rotated: groq1 → gemini1 → groq2 → gemini2
_ALL_ROTATION = []

def _build_rotation():
    """Build the rotation list from env keys."""
    global _ALL_ROTATION
    _ALL_ROTATION = []
    pairs = [
        ("groq",   os.getenv("GROQ_API_KEY_1")),
        ("gemini", os.getenv("GOOGLE_API_KEY_1")),
        ("groq",   os.getenv("GROQ_API_KEY_2")),
        ("gemini", os.getenv("GOOGLE_API_KEY_2")),
    ]
    for provider, key in pairs:
        if key and not key.startswith("your_"):
            _ALL_ROTATION.append((provider, key))

_single_call_counter = 0

def _make_llm(provider: str, api_key: str, temperature: float):
    """Create an LLM instance for the given provider."""
    if provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=api_key,
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=temperature,
            google_api_key=api_key,
        )

def invoke_with_fallback(prompt, temperature: float = 0.7):
    """Invoke a single-shot LLM call with automatic fallback on rate limit."""
    global _single_call_counter

    if not _ALL_ROTATION:
        _build_rotation()

    attempts = len(_ALL_ROTATION)
    last_error = None

    for _ in range(attempts):
        provider, api_key = _ALL_ROTATION[_single_call_counter % len(_ALL_ROTATION)]
        _single_call_counter += 1
        try:
            llm = _make_llm(provider, api_key, temperature)
            logger.info(f"Trying {provider} (key ...{api_key[-6:]})")
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in str(e) or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                logger.warning(f"{provider} rate limited, trying next provider...")
                continue
            raise  # non-rate-limit error, re-raise immediately

    raise last_error  # all providers exhausted

_build_rotation()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("csv_agent_react.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Vesco Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI styling — force light theme + indigo/emerald palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== FORCE LIGHT THEME (overrides dark mode) ===== */
    .stApp {
        background-color: #f9fafb !important;
        color: #111827 !important;
    }
    .stApp p, .stApp span, .stApp li, .stApp td, .stApp th,
    .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown, [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p {
        color: #111827 !important;
    }
    .stApp a { color: #4f46e5 !important; }
    .stApp code { color: #1f2937 !important; background: #f3f4f6 !important; }

    /* Sidebar — fixed width, no scrolling, light background */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        color: #111827 !important;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.25rem 0.6rem 0.5rem 0.6rem !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        scrollbar-width: none !important;       /* Firefox */
        -ms-overflow-style: none !important;    /* IE/Edge */
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        display: none !important;               /* Chrome/Safari */
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.3rem 0 !important;
        border-color: #e5e7eb !important;
    }
    /* Prevent sidebar resize handle */
    section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"],
    section[data-testid="stSidebar"] .resize-handle,
    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
        pointer-events: none !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar buttons — compact text-link style, LEFT aligned */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.7rem !important;
        color: #374151 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 5px !important;
        padding: 0.2rem 0.5rem !important;
        min-height: 0 !important;
        line-height: 1.3 !important;
    }
    section[data-testid="stSidebar"] .stButton > button p,
    section[data-testid="stSidebar"] .stButton > button span,
    section[data-testid="stSidebar"] .stButton > button div {
        text-align: left !important;
        color: inherit !important;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #eef2ff !important;
        color: #4338ca !important;
        border-color: #e0e7ff !important;
    }

    /* ===== Assistant chat messages — white card with BLACK text ===== */
    [data-testid="stChatMessage"] {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        color: #111827 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #111827 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: white !important;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        color: #374151 !important;
        background: transparent !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: white !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: white !important;
    }

    /* Status / progress — don't override internal layout */
    [data-testid="stStatus"] {
        border-radius: 12px;
    }
    [data-testid="stStatus"] p,
    [data-testid="stStatus"] span,
    [data-testid="stStatus"] label {
        color: #111827 !important;
    }

    /* Constrain chart size */
    .stPlotlyChart, .element-container:has(canvas) {
        max-width: 650px !important;
    }

    /* ===== User message bubble (right-aligned, indigo) ===== */
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-bottom: 0.75rem;
        padding: 0 0.5rem;
    }
    .user-bubble .bubble {
        background: #4f46e5 !important;
        color: white !important;
        padding: 10px 16px;
        border-radius: 16px 16px 4px 16px;
        max-width: 70%;
        line-height: 1.55;
        font-size: 0.88rem;
        word-wrap: break-word;
    }
    .user-bubble .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #4f46e5;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        flex-shrink: 0;
        font-size: 0.8rem;
    }

    /* ===== Dataset card (compact) ===== */
    .ds-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 3px 6px;
        background: #f9fafb !important;
        border-radius: 5px;
        border: 1px solid #e5e7eb;
        margin-bottom: 3px;
        font-size: 0.65rem;
    }
    .ds-card .ds-name {
        display: flex;
        align-items: center;
        gap: 4px;
        color: #374151 !important;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 160px;
    }
    .ds-card .ds-rows {
        color: #6b7280 !important;
        font-size: 0.58rem;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ===== Reduce spacing in chat area ===== */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem !important;
    }

    /* Chat input area — tight spacing */
    [data-testid="stBottom"] {
        background: #f9fafb !important;
    }
    [data-testid="stBottom"] > div {
        gap: 0 !important;
        padding-bottom: 2px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #111827 !important;
        background: white !important;
    }

    /* ===== Footer pinned right below chat input ===== */
    [data-testid="stBottom"] > div::after {
        content: "Powered by Vesco  •  AI can make mistakes. Verify important numbers.";
        display: block;
        text-align: center;
        font-size: 0.68rem;
        color: #9ca3af !important;
        padding: 2px 0 0 0;
        margin-top: 0;
    }

    /* Expander / selectbox in sidebar — force light */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Token Management Utilities
# =============================================================================

def extract_text(content) -> str:
    """Extract plain text from Claude's content response.
    Handles both string and list-of-blocks formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def estimate_tokens(text: str) -> int:
    """Rough estimation: 1 token ≈ 4 characters for English text."""
    return len(text) // 4

def truncate_text(text: str, max_tokens: int = 2000) -> str:
    """Truncate text to approximately max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return text
    
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    
    keep_chars = max_chars // 2
    return (
        text[:keep_chars] + 
        f"\n\n... [truncated {len(text) - max_chars} characters] ...\n\n" + 
        text[-keep_chars:]
    )

# =============================================================================
# Data loading from data/ folder
# =============================================================================

@st.cache_data
def load_csv_file(filepath: Path) -> Dict[str, pd.DataFrame]:
    """Load a CSV file."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        return {"data": df}
    except Exception as e:
        logger.error(f"Failed to load CSV {filepath}: {e}", exc_info=True)
        return {}

@st.cache_data
def load_excel_file(filepath: Path) -> Dict[str, pd.DataFrame]:
    """Load all sheets from an Excel file."""
    try:
        excel_file = pd.ExcelFile(filepath)
        sheets = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str)
            sheets[sheet_name] = df
        return sheets
    except Exception as e:
        logger.error(f"Failed to load Excel {filepath}: {e}", exc_info=True)
        return {}

def load_all_data_from_folder(data_folder: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Auto-load all CSV and Excel files from the specified folder.
    Returns dict with keys like 'filename.csv' or 'filename.xlsx::SheetName'
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.warning(f"Data folder '{data_folder}' does not exist. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        return {}
    
    combined = {}
    
    csv_files = list(data_path.glob("*.csv"))
    excel_files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls"))
    
    logger.info(f"Found {len(csv_files)} CSV files and {len(excel_files)} Excel files in {data_folder}/")
    
    for filepath in csv_files:
        out = load_csv_file(filepath)
        if out:
            combined[filepath.name] = out["data"]
            logger.info(f"Loaded CSV: {filepath.name}")
    
    for filepath in excel_files:
        out = load_excel_file(filepath)
        for sheet_name, df in out.items():
            key = f"{filepath.name}::{sheet_name}"
            combined[key] = df
            logger.info(f"Loaded Excel sheet: {key}")
    
    return combined


def build_data_schema(dfs_dict: Dict[str, pd.DataFrame]) -> str:
    """Build a compact schema string for all datasets.
    Gives the LLM column names, types, and sample values so it can skip exploration."""
    parts = []
    for key, df in dfs_dict.items():
        cols = []
        for col in df.columns[:25]:  # cap at 25 columns
            dtype = str(df[col].dtype)
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    cols.append(f"  {col} ({dtype}): min={non_null.min()}, max={non_null.max()}")
                else:
                    cols.append(f"  {col} ({dtype}): all null")
            elif df[col].dtype == "object":
                samples = df[col].dropna().unique()[:4]
                samples = [str(s)[:30] for s in samples]
                cols.append(f"  {col} (str): {samples}")
            else:
                cols.append(f"  {col} ({dtype})")
        parts.append(f"Dataset '{key}' ({len(df)} rows, {len(df.columns)} cols):\n" + "\n".join(cols))
    return "\n\n".join(parts)


# =============================================================================
# Chart Detection and Generation (combined — single LLM call)
# =============================================================================

def detect_and_generate_chart(user_question: str, answer_text: str, dfs_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Single LLM call: decide if chart is needed AND generate code if yes.
    Returns chart Python code string or None."""
    try:
        dataset_keys = list(dfs_dict.keys())
        dataset_info = []
        for key, df in dfs_dict.items():
            dataset_info.append(f"  - '{key}': {len(df)} rows, columns = {df.columns.tolist()}")

        prompt = f"""You are a data visualization assistant. Given a question and its answer, decide if a chart helps AND generate the code in ONE response.

Question: {user_question}
Answer: {answer_text[:1500]}

DATASETS in `dfs` dict:
{chr(10).join(dataset_info)}

RULES:
- If the answer is purely textual, yes/no, or a single number → reply exactly: NO
- If the answer has numerical comparisons, rankings, trends, or distributions → generate chart code

If chart IS needed, reply with ONLY a Python code block (no other text):
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
df = dfs['{dataset_keys[0] if dataset_keys else "key"}']
# ... data processing matching the answer above ...
fig, ax = plt.subplots(figsize=(8, 4.5))
# ... plot with value labels ...
ax.set_title('...', fontsize=12, fontweight='bold')
plt.tight_layout()
```

If NO chart needed, reply exactly: NO"""

        response = extract_text(invoke_with_fallback(prompt, temperature=0.2).content).strip()
        logger.info(f"Chart detect+gen response length: {len(response)}")

        if "NO" in response.upper() and "```" not in response:
            return None

        # Extract code
        if "```python" in response:
            code = response[response.index("```python") + 9:response.rindex("```")].strip()
            return code
        elif "```" in response:
            code = response[response.index("```") + 3:response.rindex("```")].strip()
            return code

        return None

    except Exception as e:
        logger.error(f"Chart detect+gen error: {str(e)}", exc_info=True)
        return None


def execute_chart_code(code: str, dfs_dict: Dict[str, pd.DataFrame]) -> Optional[plt.Figure]:
    """
    Execute the generated chart code and return the figure object.
    """
    try:
        # Create execution environment
        exec_globals = {
            'dfs': dfs_dict,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'fig': None
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Get the figure
        fig = exec_globals.get('fig')
        if fig is None:
            # Try to get current figure
            fig = plt.gcf()
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart execution error: {str(e)}")
        logger.error(f"Failed chart code:\n{code}")
        logger.error(f"Available dataset keys: {list(dfs_dict.keys())}")
        return None


# =============================================================================
# ReAct agent with PythonREPL
# =============================================================================

def build_python_repl_tool(dfs_dict: Dict[str, pd.DataFrame]):
    """Create a Python REPL tool with all DataFrames in `dfs` (dict: name -> DataFrame).
    Tool is named run_dataframe_code to avoid collision with built-in 'python' tool names."""
    repl = PythonREPL()
    repl.globals["dfs"] = dfs_dict
    repl.globals["pd"] = pd
    keys = list(dfs_dict.keys())
    if len(keys) == 1:
        repl.globals["df"] = dfs_dict[keys[0]]

    @tool
    def run_dataframe_code(code: str) -> str:
        """Execute Python code. DataFrames are in dict `dfs` (e.g. dfs['file.csv'], dfs['report.xlsx::Sheet1']). Use print() to show output."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        
        result_str = str(result)
        if len(result_str) > 4000:
            result_str = result_str[:2000] + f"\n... [truncated] ..." + result_str[-2000:]
        
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result_str}\n\nIf done, respond with FINAL ANSWER."
    
    run_dataframe_code.name = "run_dataframe_code"
    keys_preview = ", ".join(keys[:8]) + ("..." if len(keys) > 8 else "")
    run_dataframe_code.description = f"Run Python code. DataFrames in `dfs`: keys = {keys_preview}. Use dfs['key'] to get a DataFrame. Use print() to show values."
    return run_dataframe_code


def get_agent_instructions(schema: str) -> str:
    """Agent instructions with embedded data schema + formatting rules."""
    return f"""You are a business metrics analyst for an Indian distribution company. Extract EXACT numbers from the data.

TOOL: run_dataframe_code — execute Python code. Use print() to show output.

DATA SCHEMA (use these exact column names — do NOT run df.columns or df.head to explore):
{schema}

Access: dfs['key'] for each dataset. If only one dataset, `df` is also available.

RULES:
1. Write & execute code to get numbers — never guess
2. Use groupby/agg/value_counts/sort_values for accurate results
3. For dates: convert Excel serial dates → pd.to_datetime(..., unit='D', origin='1899-12-30')
4. If error, fix and retry
5. Do not create fake data

RESPONSE FORMAT (your final answer must follow this exactly):
- Start with 2-3 sentence insight answering the question with EXACT numbers
- If multiple items (rankings, comparisons, breakdowns), add a markdown table
- Use ₹ for currency values
- No technical jargon (no "groupby", "dataframe", "aggregation")
- Keep it concise — max 1-2 insights"""


def get_tool_calling_prompt(schema: str) -> ChatPromptTemplate:
    """Prompt for tool-calling agent with data schema baked in."""
    return ChatPromptTemplate.from_messages([
        ("system", get_agent_instructions(schema)),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])


def _build_executor_for_llm(llm, dfs_dict: Dict[str, pd.DataFrame], schema: str) -> AgentExecutor:
    """Build an AgentExecutor with data schema baked into the prompt."""
    tools = [build_python_repl_tool(dfs_dict)]
    prompt = get_tool_calling_prompt(schema)
    llm_with_tools = llm.bind_tools(tools)
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def create_agent_executor():
    """Return the rotation list for run_agent to try each provider."""
    if not _ALL_ROTATION:
        return None, "No API keys found. Set GROQ_API_KEY_1/2 and GOOGLE_API_KEY_1/2 in .env file."
    return _ALL_ROTATION, None


def run_agent(rotation_list: list, dfs_dict: Dict[str, pd.DataFrame], user_input: str) -> str:
    """Try each provider in rotation until one succeeds.
    Builds a fresh executor per attempt so a bad key doesn't block everything."""
    last_error = None
    schema = build_data_schema(dfs_dict)

    for provider, api_key in rotation_list:
        try:
            logger.info(f"Agent attempt: {provider} (key ...{api_key[-6:]})")
            llm = _make_llm(provider, api_key, temperature=0.7)
            executor = _build_executor_for_llm(llm, dfs_dict, schema)

            result = executor.invoke({
                "input": user_input,
            })

            output = extract_text(result.get("output", str(result)))
            if len(output) > 8000:
                output = truncate_text(output, max_tokens=2000)
            return output

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            logger.warning(f"Agent failed with {provider}: {e}")
            # If rate-limit / quota / connectivity → try next provider
            if any(kw in err_str for kw in ["429", "rate", "quota", "resource_exhausted", "retrying", "503"]):
                continue
            # For other errors, also try next provider instead of crashing
            continue

    # All providers failed
    error_msg = str(last_error) if last_error else "Unknown error"
    logger.error(f"All agent providers failed. Last error: {error_msg}")
    if "413" in error_msg or "Request too large" in error_msg:
        return "⚠️ Request too large. Try a simpler question or clear chat history."
    return "⚠️ All API providers failed. Check your API keys in .env file or try again later."


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    # =========================================================================
    # Initialize session state
    # =========================================================================
    if "chat_history_react" not in st.session_state:
        st.session_state.chat_history_react = []
    if "current_dfs" not in st.session_state:
        st.session_state.current_dfs = {}
    if "primary_key" not in st.session_state:
        st.session_state.primary_key = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # =========================================================================
    # Auto-load data (before rendering so sidebar can show dataset info)
    # =========================================================================
    if not st.session_state.data_loaded:
        combined = load_all_data_from_folder("data")
        if combined:
            st.session_state.current_dfs = combined
            st.session_state.primary_key = next(iter(combined))
            st.session_state.data_loaded = True
            logger.info(f"Auto-loaded {len(combined)} datasets")

    dfs_dict = st.session_state.current_dfs

    # =========================================================================
    # Sidebar (matches TSX layout)
    # =========================================================================
    # --- Branding (compact) ---
    with st.sidebar:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:6px; padding:2px 0 4px 0;">
            <span style="font-size:1rem;">📊</span>
            <span style="font-weight:700; color:#4338ca; font-size:0.85rem;">Vesco Analyst Pro</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- Active Datasets ---
        st.markdown("""
        <p style="font-size:0.62rem; font-weight:600; color:#6b7280;
           text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
           Active Datasets</p>
        """, unsafe_allow_html=True)

        if not st.session_state.data_loaded:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; padding:8px; color:#9ca3af; font-size:0.85rem;">
                ⏳ Loading datasets…
            </div>""", unsafe_allow_html=True)
        elif not dfs_dict:
            st.markdown("""
            <div style="padding:8px; color:#ef4444; font-size:0.8rem; font-style:italic;">
                No datasets loaded. Add files to data/ folder.
            </div>""", unsafe_allow_html=True)
        else:
            icon_map = {"visit": "👤", "sales": "📈", "order": "📄",
                        "receipt": "🗄️", "customer": "👥", "account": "👥"}
            for key, df in dfs_dict.items():
                icon = "📋"
                for keyword, emoji in icon_map.items():
                    if keyword in key.lower():
                        icon = emoji
                        break
                display_name = (key.replace("::", " › ")
                                   .replace(".xlsx", "")
                                   .replace(".csv", "")
                                   .replace(".xls", ""))
                st.markdown(f"""
                <div class="ds-card">
                    <div class="ds-name"><span>{icon}</span><span>{display_name}</span></div>
                    <span class="ds-rows">{df.shape[0]:,} rows</span>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Quick Analysis (suggested queries) ---
        st.markdown("""
        <p style="font-size:0.62rem; font-weight:600; color:#6b7280;
           text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
           Quick Analysis</p>
        """, unsafe_allow_html=True)

        suggested_queries = [
            "How many visits did Ajay do in November?",
            "Top 5 customers by sales",
            "Which salesman has worst collection efficiency?",
            "Show me the aging summary for Krishna Cycle Stores"
            "What is the visit conversion rate for Nitin?",
        ]
        for i, query in enumerate(suggested_queries):
            if st.button(query, key=f"sq_{i}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()

        st.divider()

        # --- Actions ---
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("🔄 Reload", key="btn_reload", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.current_dfs = {}
                st.session_state.primary_key = None
                st.rerun()
        with act_col2:
            if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
                st.session_state.chat_history_react = []
                st.rerun()

        # --- Status footer (compact) ---
        status_label = ("Online" if st.session_state.data_loaded and dfs_dict
                        else "Loading…" if not st.session_state.data_loaded
                        else "No data")
        ds_count = len(dfs_dict)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; padding:6px 0 2px 0;">
            <div style="width:26px; height:26px; border-radius:50%; background:#e0e7ff;
                 display:flex; align-items:center; justify-content:center;
                 color:#4f46e5; font-weight:700; font-size:0.6rem;">AI</div>
            <div>
                <div style="font-size:0.78rem; font-weight:500; color:#111827;">Vesco AI</div>
                <div style="font-size:0.68rem; color:#6b7280;">{status_label} • {ds_count} datasets</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # Main content area
    # =========================================================================

    # No-data screen
    if not dfs_dict and st.session_state.data_loaded:
        st.markdown("""
        <div style="background:#eff6ff; border:1px solid #93c5fd; border-radius:12px;
             padding:2rem; margin:2rem auto; max-width:600px; text-align:center;">
            <h3 style="margin-top:0; color:#1e3a5f;">📁 No data found</h3>
            <p style="color:#4b5563;">
                Add CSV or Excel files to the <code>data/</code> folder and click
                <strong>Reload</strong> in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.data_loaded:
        st.info("⏳ Loading datasets from data/ folder…")
        return

    # --- Helper: right-aligned user bubble ---
    def _user_bubble(text: str):
        safe = html_lib.escape(text)
        st.markdown(f'''
        <div class="user-bubble">
            <div class="bubble">{safe}</div>
            <div class="avatar">👤</div>
        </div>''', unsafe_allow_html=True)

    # ---- Tabs: Agent Chat | Data Preview ----
    tab_chat, tab_data = st.tabs(["🤖 Agent Chat", "📊 Data Preview"])

    with tab_chat:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history_react:
                with st.chat_message("assistant", avatar="📊"):
                    datasets_info = "\n".join([
                        f"- **{k}**: {v.shape[0]:,} rows, {v.shape[1]} columns"
                        for k, v in dfs_dict.items()
                    ])
                    st.markdown(
                        f"**Hello! I am your Business Intelligence Analyst.**\n\n"
                        f"Data loaded successfully!\n\n{datasets_info}\n\n"
                        f"How can I help you analyze this data?"
                    )
            else:
                for msg in st.session_state.chat_history_react:
                    if msg["role"] == "user":
                        _user_bubble(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="📊"):
                            st.markdown(msg.get("content", ""))
                            if "chart" in msg and msg["chart"] is not None:
                                col_l, col_c, col_r = st.columns([0.5, 2, 0.5])
                                with col_c:
                                    st.pyplot(msg["chart"])

    with tab_data:
        if dfs_dict:
            primary_key = st.session_state.primary_key or next(iter(dfs_dict))
            if len(dfs_dict) > 1:
                selected_key = st.selectbox(
                    "Select dataset",
                    options=list(dfs_dict.keys()),
                    index=list(dfs_dict.keys()).index(primary_key) if primary_key in dfs_dict else 0,
                    key="data_preview_select",
                )
                if selected_key != primary_key:
                    st.session_state.primary_key = selected_key
                    primary_key = selected_key
            preview_df = dfs_dict[primary_key]
            st.dataframe(preview_df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 rows · Total: {preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")
        else:
            st.info("No datasets loaded.")

    # --- Input handling (always visible, outside tabs) ---
    chat_input_value = st.chat_input("Ask a question about sales, visits, or accounts…")

    # Pick up a pending query from sidebar suggested buttons
    user_input = chat_input_value
    if not user_input and st.session_state.get("pending_query"):
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        if not dfs_dict:
            st.error("No data loaded. Add files to data/ folder and reload.")
            return

        if not _ALL_ROTATION:
            st.error("⚠️ No API keys configured. Set GROQ_API_KEY_1/2 and GOOGLE_API_KEY_1/2 in .env file.")
            return

        rotation_list, err = create_agent_executor()
        if err:
            st.error(f"⚠️ {err}")
            return

        # Add user message to history
        st.session_state.chat_history_react.append({
            "role": "user",
            "content": user_input,
        })

        # Show user message immediately
        _user_bubble(user_input)

        # Process with progress indicator
        with st.status("Analyzing your data…", expanded=True) as status:
            progress = st.progress(0, text="Starting analysis…")

            progress.progress(15, text="Querying data and running code…")
            formatted_output = run_agent(rotation_list, dfs_dict, user_input)
            progress.progress(70, text="Checking if a chart would be helpful…")

            chart_fig = None
            chart_code = detect_and_generate_chart(user_input, formatted_output, dfs_dict)
            if chart_code:
                progress.progress(90, text="Generating chart…")
                chart_fig = execute_chart_code(chart_code, dfs_dict)

            progress.progress(100, text="Done!")
            status.update(label="Analysis complete", state="complete", expanded=False)

        # Save assistant response
        st.session_state.chat_history_react.append({
            "role": "assistant",
            "content": formatted_output,
            "chart": chart_fig,
        })

        st.rerun()


if __name__ == "__main__":
    main()