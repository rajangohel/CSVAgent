"""
CSV Agent (ReAct + REPL) - AI-Powered CSV Analysis with Chart Visualization
Auto-loads all Excel/CSV files from data/ folder
Implements: Context Creation → Prompt Augmentation → Code Generation → Execution → Chart Generation
Uses LangChain ReAct agent with PythonREPL for safe code execution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import io
import base64

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain_anthropic import ChatAnthropic

load_dotenv()

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
    page_title="Vesco AI - Analyze & Visualize",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0d9488 0%, #0f766e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chart-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .data-folder-info {
        background: #eff6ff;
        border: 1px solid #93c5fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .chart-suggestion-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    /* Constrain chart size */
    .stPlotlyChart, .element-container:has(canvas) {
        max-width: 650px !important;
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

def truncate_chat_history(chat_history: list, max_messages: int = 6, max_tokens_per_msg: int = 500) -> str:
    """Truncate chat history to prevent token overflow."""
    recent_messages = chat_history[-max_messages:]
    
    formatted_messages = []
    for msg in recent_messages:
        role = msg['role']
        content = msg.get('content', '')
        
        if estimate_tokens(content) > max_tokens_per_msg:
            content = truncate_text(content, max_tokens_per_msg)
        
        formatted_messages.append(f"{role}: {content}")
    
    return "\n".join(formatted_messages)


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


# =============================================================================
# Chart Detection and Generation
# =============================================================================

def detect_chart_opportunity(user_question: str, answer_text: str) -> Optional[Dict[str, Any]]:
    """
    Decide whether a chart makes sense for this Q&A.
    Returns None or a dict describing the chart idea.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            api_key=api_key,
        )

        prompt = f"""You are deciding whether a chart would meaningfully support the answer.

Question: {user_question}

Answer: {answer_text}

Rules:
- If the answer contains NUMERICAL data that would benefit from visualization (comparisons, trends, distributions, rankings), suggest a chart
- If the answer is purely textual, descriptive, or a simple yes/no, reply exactly: NO
- Chart types: bar_chart, line_chart, pie_chart, scatter_plot, histogram

If a chart WOULD help, reply ONLY with valid JSON (no extra text):
{{
  "chart_type": "bar_chart or line_chart or pie_chart or scatter_plot or histogram",
  "data_description": "what data to visualize",
  "reason": "brief reason (max 15 words)"
}}

If NO chart needed, reply exactly: NO"""

        response = extract_text(llm.invoke(prompt).content).strip()
        logger.info(f"Chart detection response: {response}")

        if "NO" in response.upper() and "{" not in response:
            return None

        # Try to extract JSON from response
        if "{" in response:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            json_str = response[json_start:json_end]
            chart_data = json.loads(json_str)
            return chart_data
        
        return None
        
    except Exception as e:
        logger.error(f"Chart detection error: {str(e)}", exc_info=True)
        return None


def generate_chart_code(user_question: str, chart_intent: Dict[str, Any], dfs_dict: Dict[str, pd.DataFrame], raw_data: str = None) -> Optional[str]:
    """
    Generate Python code to create the chart based on the intent.
    Returns Python code as string or None if generation fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,  # Lower temperature for more deterministic code generation
            api_key=api_key,
        )

        # Get COMPLETE dataset info - all datasets with ALL columns
        dataset_info = []
        for key, df in dfs_dict.items():
            # Show ALL columns, not just first few
            all_columns = df.columns.tolist()
            dataset_info.append(f"  - '{key}': {len(df)} rows, columns = {all_columns}")

        # Get more context from raw_data
        raw_data_context = ""
        if raw_data:
            # Include more of the analysis result (up to 1500 chars instead of 500)
            raw_data_context = f"\nAgent's exact analysis result:\n{raw_data[:1500]}"

        # List all dataset keys explicitly
        dataset_keys = list(dfs_dict.keys())
        
        prompt = f"""Generate Python code to create a {chart_intent['chart_type']} for this question.

Question: {user_question}
Data to visualize: {chart_intent.get('data_description', '')}
{raw_data_context}

AVAILABLE DATASETS in `dfs` dictionary - USE THESE EXACT KEYS:
{chr(10).join(dataset_info)}

CRITICAL RULES:
1. You MUST use dfs['EXACT_KEY_FROM_ABOVE'] - do NOT make up dataset names
2. First dataset key to use: '{dataset_keys[0] if dataset_keys else 'unknown'}'
3. Use ONLY the column names listed above - do NOT invent column names
4. Copy the EXACT data processing logic from the analysis result above
5. If analysis used filtering/groupby/aggregation, use THE EXACT SAME code

Code Requirements:
- Use matplotlib for visualization
- Create figure with figsize=(8, 4.5)
- Add value labels on bars/points showing exact numbers
- Use plt.tight_layout()
- Store figure in variable named 'fig'
- Use seaborn style: sns.set_style("whitegrid")

Generate ONLY the Python code, no explanations:

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Get data using EXACT key from dfs dict
df = dfs['{dataset_keys[0] if dataset_keys else 'key'}']  # MUST use exact key

# Data processing - copy from analysis above
# ...

# Create plot
fig, ax = plt.subplots(figsize=(8, 4.5))

# Plot data and add value labels
# bars = ax.bar(x, y)
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height,
#             f'{{int(height)}}', ha='center', va='bottom', fontsize=9)

ax.set_title('Title', fontsize=12, fontweight='bold')
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
plt.tight_layout()
```

Generate the complete code now:"""

        response = extract_text(llm.invoke(prompt).content).strip()
        logger.info(f"Generated chart code length: {len(response)}")

        # Extract code from markdown if present
        if "```python" in response:
            code_start = response.index("```python") + 9
            code_end = response.rindex("```")
            code = response[code_start:code_end].strip()
        elif "```" in response:
            code_start = response.index("```") + 3
            code_end = response.rindex("```")
            code = response[code_start:code_end].strip()
        else:
            code = response

        return code

    except Exception as e:
        logger.error(f"Chart code generation error: {str(e)}", exc_info=True)
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


def get_agent_instructions() -> str:
    """Agent instructions for tool-calling agent."""
    return """You are a precise business metrics analyst. Your only job is to extract EXACT numbers and calculate meaningful ratios/trends from the data.

You have exactly one tool: run_dataframe_code. Use only this tool to run Python code. Do not request any other code execution or python tool.

All loaded datasets are in the dict `dfs` — do not read files again.
- Keys are file names (e.g. 'sales.csv') or 'filename::SheetName' for Excel sheets (e.g. 'report.xlsx::Q1').
- Use dfs['key'] to get a DataFrame. Example: df1 = dfs['data.csv']; df2 = dfs['report.xlsx::Sheet1'].
- If only one dataset exists, it is also in variable `df` for convenience.

Rule – MUST follow:
1. ALWAYS write & execute code to get numbers — never guess/estimate
2. Use groupby/agg/value_counts/sort_values to get accurate counts, sums, averages
3. Calculate percentages, growth rates, top-N when relevant
4. For dates: convert Excel serial dates (e.g. 45903) → pd.to_datetime(..., unit='D', origin='1899-12-30')
5. Show concise output: .head(6), .tail(3), value_counts(), describe(), or aggregated tables
6. If you get an error, fix the code and try again.
7. Prefer running code to get the answer rather than guessing.
8. If you cannot answer with code, reply "I don't know".
9. Do not create example or fake dataframes.
10. When asked for chart data, produce the aggregated/transformed data (e.g. as dict or list) suitable for plotting; use print() so the result is visible.
11. Final Answer must be short, number-heavy, analytical (1–2 insights max)"""


def get_tool_calling_prompt() -> ChatPromptTemplate:
    """Prompt for tool-calling agent (native tool use; works with Claude)."""
    return ChatPromptTemplate.from_messages([
        ("system", get_agent_instructions()),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])


def create_agent_executor(dfs_dict: Dict[str, pd.DataFrame]):
    """Create tool-calling agent with Python REPL and Claude LLM."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "Set ANTHROPIC_API_KEY in .env file to use the agent."

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        api_key=api_key,
    )

    tools = [build_python_repl_tool(dfs_dict)]
    prompt = get_tool_calling_prompt()
    llm_with_tools = llm.bind_tools(tools)
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return executor, None


def format_agent_output(user_question: str, raw_output: str, dfs_dict: Dict[str, pd.DataFrame], executor: AgentExecutor = None) -> tuple:
    """
    Format the agent output to be more user-friendly.
    Returns: (formatted_text, data_dict) where data_dict contains the actual values for chart validation
    """
    try:
        final_answer = raw_output.strip()
        
        # Try to extract numerical data from the raw output for validation
        data_dict = {}
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return raw_output, data_dict

        llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.3,
            api_key=api_key,
        )
        
        data_context = []
        for key, df in list(dfs_dict.items())[:3]:
            data_context.append(f"- {key}: {df.shape[0]} rows, columns: {', '.join(df.columns.tolist()[:5])}")
        
        insight_prompt = f"""You are a friendly data analyst explaining results to a non-technical user.

User asked: {user_question}
Raw analysis result: {final_answer}
Datasets used:
{chr(10).join(data_context)}

Write a 3-5 line human-readable response with this structure:
Line 1: Directly answer the user's question with the exact numbers found.
Line 2: Briefly mention what approach was used to get the answer (e.g. "filtered by X, grouped by Y, summed Z") — keep it simple, no code.
Line 3-5: A short reasoning or insight explaining what the numbers mean or why they matter.

Rules:
- Write in plain conversational English, like talking to a colleague
- Use the EXACT numbers from the raw result — never round or guess
- No bullet points, no headings, no markdown formatting
- No technical jargon (no "groupby", "aggregation", "dataframe")
- Keep the total response under 5 sentences

Example:
"The total sales for March came to ₹12,45,000 across 47 orders. This was calculated by looking at all completed orders in March and adding up their values. That's actually a 12% jump compared to February's ₹11,12,000, which suggests the new campaign is driving results. Rajan led with ₹3,20,000 from 15 deals, making him the top performer for the month."

Response:"""

        try:
            insights_response = llm.invoke(insight_prompt)
            insights = extract_text(insights_response.content).strip()
            return insights, data_dict
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return final_answer, data_dict
        
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        return raw_output, {}


def run_agent(agent_executor: AgentExecutor, user_input: str, chat_history_list: list = None) -> str:
    """Invoke agent and return final output."""
    try:
        # Convert chat history to proper message objects for MessagesPlaceholder
        chat_messages = []
        if chat_history_list:
            # Take last 5 messages to avoid token overflow
            recent_history = chat_history_list[-10:]  # Get last 10 messages (5 exchanges)
            for msg in recent_history:
                content = msg.get('content', '')
                # Truncate long messages
                if estimate_tokens(content) > 400:
                    content = truncate_text(content, max_tokens=400)
                
                if msg['role'] == 'user':
                    chat_messages.append(HumanMessage(content=content))
                elif msg['role'] == 'assistant':
                    chat_messages.append(AIMessage(content=content))
        
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_messages,  # Pass as list of message objects
        })
        
        output = extract_text(result.get("output", str(result)))

        if len(output) > 8000:
            output = truncate_text(output, max_tokens=2000)
        
        return output
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Agent execution failed: {error_msg}", exc_info=True)
        
        if "413" in error_msg or "Request too large" in error_msg:
            return "⚠️ Request too large. Try a simpler question or clear chat history."
        elif "rate_limit" in error_msg.lower():
            return "⚠️ Rate limit exceeded. Please wait and try again."
        else:
            return "⚠️ Something went wrong. Please try again or rephrase your question."


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Vesco AI </h1>', unsafe_allow_html=True)

    # Initialize session state
    if "chat_history_react" not in st.session_state:
        st.session_state.chat_history_react = []
    if "current_dfs" not in st.session_state:
        st.session_state.current_dfs = {}
    if "primary_key" not in st.session_state:
        st.session_state.primary_key = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    # Auto-load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data from data/ folder..."):
            combined = load_all_data_from_folder("data")
            if combined:
                st.session_state.current_dfs = combined
                st.session_state.primary_key = next(iter(combined))
                st.session_state.data_loaded = True
                logger.info(f"Auto-loaded {len(combined)} datasets")

    dfs_dict = st.session_state.current_dfs

    # No data screen
    if not dfs_dict:
        st.markdown('<div class="data-folder-info">', unsafe_allow_html=True)
        st.markdown("### 📁 No data found in `data/` folder")
        st.markdown("""
        **To get started:**
        1. Create a `data/` folder in the same directory as this script
        2. Add your CSV or Excel files
        3. Refresh this page
        
        **Supported formats:** CSV (`.csv`), Excel (`.xlsx`, `.xls`)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Reload data"):
            st.session_state.data_loaded = False
            st.rerun()
        
        return

    # Main interface
    primary_key = st.session_state.primary_key
    df = dfs_dict[primary_key]

    # Top controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.success(f"✅ Loaded {len(dfs_dict)} dataset(s)")
    
    with col2:
        if st.button("🔄 Reload"):
            st.session_state.data_loaded = False
            st.session_state.current_dfs = {}
            st.session_state.primary_key = None
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history_react = []
            st.rerun()

    st.divider()

    # Tabs
    tabs = st.tabs(["🤖 Agent Chat", "📊 Data Preview"])

    with tabs[0]:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history_react:
                for i, msg in enumerate(st.session_state.chat_history_react):
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(msg.get("content", ""))
                            
                            # Show chart if available - smaller size
                            if "chart" in msg and msg["chart"] is not None:
                                col1, col2, col3 = st.columns([0.5, 2, 0.5])
                                with col2:
                                    st.pyplot(msg["chart"], width='content')
            else:
                st.info("👋 Start a conversation! Ask me anything about your data.")
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your data...")
        
        if user_input:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("⚠️ ANTHROPIC_API_KEY not configured.")
            else:
                executor, err = create_agent_executor(dfs_dict)
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    # Add user message
                    st.session_state.chat_history_react.append({
                        "role": "user",
                        "content": user_input
                    })

                    # Show user question immediately in chat
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    # Show progress during analysis
                    with st.status("Analyzing...", expanded=True) as status:
                        progress = st.progress(0, text="Starting analysis...")

                        # Step 1: Run agent (0% -> 50%)
                        progress.progress(10, text="Querying data and running code...")
                        raw_output = run_agent(
                            executor,
                            user_input,
                            [m for m in st.session_state.chat_history_react[:-1] if m["role"] in ["user", "assistant"]]
                        )
                        progress.progress(50, text="Data analysis complete. Formatting response...")

                        # Step 2: Format output (50% -> 75%)
                        formatted_output, data_dict = format_agent_output(user_question=user_input, raw_output=raw_output, dfs_dict=dfs_dict, executor=executor)
                        progress.progress(75, text="Checking if a chart would be helpful...")

                        # Step 3: Chart detection & generation (75% -> 100%)
                        chart_intent = detect_chart_opportunity(user_input, raw_output)

                        chart_fig = None
                        if chart_intent:
                            progress.progress(85, text="Generating chart...")
                            chart_code = generate_chart_code(user_input, chart_intent, dfs_dict, raw_data=raw_output)
                            if chart_code:
                                chart_fig = execute_chart_code(chart_code, dfs_dict)

                        progress.progress(100, text="Done!")
                        status.update(label="Analysis complete", state="complete", expanded=False)
                    
                    # Add assistant message with chart
                    st.session_state.chat_history_react.append({
                        "role": "assistant",
                        "content": formatted_output,
                        "chart": chart_fig
                    })
                    
                    st.rerun()

    with tabs[1]:
        st.subheader("Data Preview")
        
        # Dataset selector
        if len(dfs_dict) > 1:
            st.markdown("**Select dataset:**")
            selected_key = st.selectbox(
                "Dataset",
                options=list(dfs_dict.keys()),
                index=list(dfs_dict.keys()).index(primary_key) if primary_key in dfs_dict else 0,
                key="primary_key_select",
                label_visibility="collapsed",
            )
            if selected_key != primary_key:
                st.session_state.primary_key = selected_key
                df = dfs_dict[selected_key]
                st.rerun()
        
        st.caption(f"**{primary_key}**")
        st.dataframe(df.head(100), width='stretch')
        st.caption(f"Showing first 100 rows • Total: {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()