import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import decimal
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables (must run before ADK setup)
load_dotenv()

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, ToolContext

# --- CONSTANTS AND SETUP ---
APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

# Set up Session Management
session_service = InMemorySessionService()

# Ensure the correct relative path is used for the test
csv_path = "data/online_sales_dataset.csv"

# Dummy for ADK retry settings
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# --- DATA PROCESSING FUNCTIONS ---

def load_sales_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and ensure proper types."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}. Check the 'data/' folder.")

    df = pd.read_csv(csv_path, low_memory=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Ensure numeric columns
    for col in ['Quantity', 'UnitPrice', 'Discount', 'ShippingCost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)

    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values, standardize text, normalize discounts, compute revenue and returns."""
    cat_fill = {
        'CustomerID': 'Unknown',
        'WarehouseLocation': 'Unknown',
        'Description': '',
        'Category': 'Unknown',
        'SalesChannel': 'Unknown',
        'ReturnStatus': 'No Return'
    }
    for col, val in cat_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col in ['PaymentMethod', 'Country', 'ShipmentProvider', 'OrderPriority']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    if 'Discount' in df.columns:
        disc = df['Discount'].copy()
        mask_pct = (disc > 1) & (disc <= 100)
        disc.loc[mask_pct] = disc.loc[mask_pct] / 100.0
        df['Discount_norm'] = disc.clip(lower=0.0).fillna(0.0)
    else:
        df['Discount_norm'] = 0.0

    return_indicators = ['return', 'returned', 'refunded', 'rma']
    df['IsReturnFlag_text'] = df['ReturnStatus'].astype(str).str.lower().fillna('')
    df['IsReturn_by_text'] = df['IsReturnFlag_text'].apply(lambda s: any(k in s for k in return_indicators))
    df['IsReturn_by_qty'] = df['Quantity'] < 0
    df['IsReturn'] = df['IsReturn_by_text'] | df['IsReturn_by_qty']

    df['LineRevenue'] = df['Quantity'] * df['UnitPrice'] * (1.0 - df.get('Discount_norm', 0.0))
    df['NetRevenue'] = df['LineRevenue']

    df.drop(columns=['IsReturnFlag_text', 'IsReturn_by_text', 'IsReturn_by_qty'], inplace=True, errors='ignore')

    return df


def process_sales_pipeline(csv_path: str) -> pd.DataFrame:
    """Full pipeline: load + clean sales data."""
    try:
        df = load_sales_data(csv_path)
        df_clean = clean_sales_data(df)
        return df_clean
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"FATAL PROCESSING ERROR: {e}")
        return pd.DataFrame()


# Money format helper
def fmt_money(x):
    try:
        # Use decimal formatting for robust output
        return f"${float(x):,.2f}"
    except:
        return x


def build_daily_report(df_all, target_date_str, lookback_days_for_trend=28):
    """
    Returns:
      day_df: DataFrame with all transactions for target_date (date portion match)
      metrics: dict with totals, AOV, top_products, revenue_by_category/channel, return_rate, anomalies, short trend context
    """
    # parse target date as date-only
    target_dt = pd.to_datetime(target_date_str).normalize()
    # Ensure InvoiceDate is timezone-naive for date comparisions
    try:
        tzinfo = getattr(df_all['InvoiceDate'].dt, 'tz', None)
    except Exception:
        tzinfo = None

    if tzinfo is not None:
        df_all['InvoiceDate_naive'] = df_all['InvoiceDate'].dt.tz_convert(None)
    else:
        df_all['InvoiceDate_naive'] = df_all['InvoiceDate']
    df_all['InvoiceDate_date'] = pd.to_datetime(df_all['InvoiceDate_naive']).dt.normalize()

    # Subset for the exact date
    day_df = df_all[df_all['InvoiceDate_date'] == target_dt].copy().reset_index(drop=True)

    # Basic metrics
    total_revenue = float(day_df['NetRevenue'].sum()) if len(day_df) else 0.0
    num_orders = int(day_df['InvoiceNo'].nunique()) if len(day_df) else 0
    num_lines = int(len(day_df))
    avg_order_value = (total_revenue / num_orders) if num_orders else 0.0
    avg_line_value = (day_df['NetRevenue'].mean() if num_lines else 0.0)

    # top products by revenue
    top_products_series = day_df.groupby(['StockCode', 'Description'])['NetRevenue'].sum().sort_values(
        ascending=False).head(10)
    top_products_list = [{'StockCode': sc, 'Description': desc, 'revenue': float(v)}
                         for (sc, desc), v in top_products_series.items()]

    # revenue breakdowns
    revenue_by_category = day_df.groupby('Category')['NetRevenue'].sum().sort_values(ascending=False).to_dict()
    revenue_by_channel = day_df.groupby('SalesChannel')['NetRevenue'].sum().sort_values(ascending=False).to_dict()

    # top countries
    top_countries_series = (
        day_df.groupby('Country')['NetRevenue']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    top_countries_list = top_countries_series.reset_index().to_dict(orient='records')

    # return rate (by line)
    if len(day_df):
        returns_count = int(day_df['IsReturn'].sum())
        return_rate = returns_count / len(day_df)
    else:
        returns_count = 0
        return_rate = 0.0

    # anomaly detection (per-line) -- simple z-score threshold
    anomalies = []
    if len(day_df) >= 5:
        mean_line = day_df['NetRevenue'].mean()
        std_line = day_df['NetRevenue'].std(ddof=0)
        if pd.isna(std_line) or std_line == 0:
            std_line = 0.0
        thresh_upper = mean_line + 3 * std_line
        thresh_lower = mean_line - 3 * std_line
        anom_df = day_df[(day_df['NetRevenue'] > thresh_upper) | (day_df['NetRevenue'] < thresh_lower)]
        anomalies = anom_df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'NetRevenue']].to_dict(
            'records')

    # short trend context: compare revenue to previous week/day and percentage change
    start_trend = target_dt - pd.Timedelta(days=lookback_days_for_trend)
    mask_trend = (df_all['InvoiceDate_date'] >= start_trend) & (df_all['InvoiceDate_date'] <= target_dt)
    series_daily = df_all.loc[mask_trend].groupby('InvoiceDate_date')['NetRevenue'].sum().sort_index()
    trend = {}
    if not series_daily.empty:
        prev_day = target_dt - pd.Timedelta(days=1)
        prev_revenue = float(series_daily.get(prev_day, 0.0))
        prev_change_pct = ((total_revenue - prev_revenue) / prev_revenue * 100.0) if prev_revenue else None
        week_start = target_dt - pd.Timedelta(days=7)
        week_mask = (series_daily.index >= week_start) & (series_daily.index < target_dt)
        week_avg = float(series_daily.loc[week_mask].mean()) if series_daily.loc[week_mask].size else None
        week_change_pct = ((total_revenue - week_avg) / week_avg * 100.0) if week_avg not in (None, 0) else None
        trend = {
            'prev_day_revenue': prev_revenue,
            'prev_day_change_pct': prev_change_pct,
            'week_avg_before': week_avg,
            'week_change_pct': week_change_pct,
            'series_daily': series_daily.to_dict()
        }
    else:
        trend = {'series_daily': {}}

    # Adjust list formatting for money
    for item in top_products_list:
        raw = item.get('revenue', 0.0)
        if 'revenue' in item: item.pop('revenue')
        item['revenue'] = fmt_money(raw)

    for item in top_countries_list:
        raw = item.pop('NetRevenue')
        item['revenue'] = fmt_money(raw)

    # Top products by revenue -> dict {Description: revenue}
    top_products_desc_series = day_df.groupby('Description')['NetRevenue'].sum().sort_values(ascending=False).head(10)
    top_products = {desc: round(float(v), 2) for desc, v in top_products_desc_series.items()}

    # Top countries by revenue -> dict {Country: revenue}
    top_countries_series = day_df.groupby('Country')['NetRevenue'].sum().sort_values(ascending=False).head(5)
    top_countries = {country: round(float(v), 2) for country, v in top_countries_series.items()}

    metrics = {
        'date': target_dt.strftime('%Y-%m-%d'),
        'total_revenue': fmt_money(total_revenue),
        'num_orders': num_orders,
        'num_lines': num_lines,
        'avg_order_value': fmt_money(avg_order_value),
        'avg_line_value': fmt_money(avg_line_value),
        'top_products': top_products_list,
        'revenue_by_category': {k: float(v) for k, v in revenue_by_category.items()},
        'revenue_by_channel': {k: float(v) for k, v in revenue_by_channel.items()},
        'top_countries': top_countries_list,
        'returns_count': returns_count,
        'return_rate': round(return_rate, 4),
        'anomalies': anomalies,
        'trend': trend
    }

    return day_df, metrics


# --- ADK UTILITIES & HELPERS ---

async def run_session(
        runner_instance: Runner,
        user_queries: list[str] | str = None,
        session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    app_name = runner_instance.app_name

    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if type(user_queries) == str:
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\nUser > {query}")
            query = types.Content(role="user", parts=[types.Part(text=query)])
            async for event in runner_instance.run_async(
                    user_id=USER_ID, session_id=session.id, new_message=query
            ):
                if event.content and event.content.parts:
                    if (
                            event.content.parts[0].text != "None"
                            and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


def sanitize_for_json(obj):
    """
    Recursively convert an object into JSON-serializable Python primitives.
    """
    # primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    # pandas / numpy scalars
    if isinstance(obj, (np.generic,)):
        return obj.item()

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        try:
            return pd.to_datetime(obj).strftime('%Y-%m-%d')
        except Exception:
            return str(obj)

    if isinstance(obj, pd.Timedelta):
        return str(obj)

    if isinstance(obj, decimal.Decimal):
        return float(obj)

    # pandas NA
    if obj is pd.NaT or (isinstance(obj, float) and np.isnan(obj)):
        return None

    # dict: sanitize keys and values
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(k, (pd.Timestamp, datetime, date)):
                new_key = pd.to_datetime(k).strftime('%Y-%m-%d')
            elif isinstance(k, (np.generic,)):
                new_key = str(k.item())
            elif not isinstance(k, (str, int, float, bool, type(None))):
                new_key = str(k)
            else:
                new_key = k
            new[new_key] = sanitize_for_json(v)
        return new

    # list/tuple/set -> list
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # pandas Series -> dict with string keys
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())

    # pandas DataFrame -> list of records (safe)
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient='records')
        return sanitize_for_json(records)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]

    # fallback: try to cast to primitive
    try:
        return float(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


# --- GLOBAL DATA LOAD (Preload for agent tools) ---
print(f"Loading data from: {csv_path}...")
df_clean = process_sales_pipeline(csv_path)

if df_clean.empty:
    print("CRITICAL: Dataframe is empty. Agent tools will fail.")
else:
    print(f"Data loaded successfully. Total records: {len(df_clean)}")


# ----------------------------------------------------

# ensure you have a df-based runner
def run_daily_report_for_date_df(df: pd.DataFrame, date_str: str):
    if 'InvoiceDate_date' not in df.columns:
        df['InvoiceDate_date'] = pd.to_datetime(df['InvoiceDate']).dt.normalize()

    target_dt = pd.to_datetime(date_str).normalize()
    if target_dt not in df['InvoiceDate_date'].values:
        print(
            f"Warning: Target date {date_str} not found. Using data minimum date: {df['InvoiceDate_date'].min().strftime('%Y-%m-%d')}")
        target_dt = df['InvoiceDate_date'].min()
    target_date_str = target_dt.strftime('%Y-%m-%d')

    day_transactions_df, metrics = build_daily_report(df, target_date_str)
    return day_transactions_df, metrics


# ADK-friendly tool function - accepts only a date string and returns sanitized JSON-friendly dict
def daily_report_tool_for_agent(date_str: str):
    """
    Tool for the agent to retrieve daily report metrics.
    """
    try:
        target_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    except Exception as e:
        return {"status": "error", "message": f"Invalid date format: {date_str}"}

    try:
        _, metrics = run_daily_report_for_date_df(df_clean, target_date)
        metrics_safe = sanitize_for_json(metrics)
        return {"status": "success", "metrics": metrics_safe}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- AGENT DEFINITIONS ---

daily_report_agent = LlmAgent(
    name="daily_report_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are a daily report assistant.
1) Extract the date from the user's prompt.
2) Convert it to 'YYYY-MM-DD'.
3) Call daily_report_tool_for_agent(date_str).
4) Return the tool output exactly if status == 'success', otherwise explain the error.
""",
    tools=[daily_report_tool_for_agent],
)


# Executive Summary processing agent is defined here
def generate_executive_summary_for_agent(metrics_json: str):
    """
    This is a dummy tool function to satisfy the Orchestrator's required tool call.
    The SummaryAgent will actually do the work.
    """
    return {"status": "success", "summary_request_received": True, "metrics_json_length": len(metrics_json)}


summary_agent = LlmAgent(
    name="summary_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are an executive summarizer. Your job is to call the prior agent to get the sanitized metrics, then turn those metrics into a single, readable executive paragraph the user can act on.

Agent behavior and process
1) Obtain metrics
   - Call the daily_report_agent tool and get back the sanitized metrics object.
   - Treat that metrics object as authoritative. Do not invent values.

2) What to read from metrics (use when present)
   - date / report_date
   - total_revenue
   - num_orders
   - avg_order_value or avg_order_value
   - revenue_by_category (dict)
   - revenue_by_channel (dict)
   - top_products (dict) pick top 3 by value
   - top_countries (dict) pick top 3 by value
   - anomalies (dict) use count and up to 2 example keys
   - trend (dict) use _prev_day_change_pct and _week_change_pct when present

3) Tone and style
   - Write in plain, engaging English aimed at a manager.
   - Use varied sentence openers and short sentences for impact.
   - Avoid dry lists. Merge related facts into compact sentences.
   - If the report contains surprising or missing numbers (for example many top product revenues are zero) call that out briefly as a data quality note. Do not speculate on causes.

4) Formatting rules
   - Produce exactly one paragraph of 4 to 6 sentences.
   - Round monetary values to two decimals and format with commas, for example $12,345.67.
   - Format percentages to two decimals and append %, for example 12.34%.
   - When stating a percentage change, also include the absolute reference if available, for example "up 15.9% (from $20,000 to $23,180)". If the absolute figure is not available, give the percent alone.

5) Sentence structure (recommended, not template locked)
   - Sentence 1: Headline. Date, total revenue, orders. Short and punchy.
     Example: "Executive summary for 2020-01-11: total revenue was $25,497.17 from 24 orders."
   - Sentence 2: Channel and AOV highlight. Mention top channel and AOV.
     Example: "Average order value was $1,062.38, with Online leading at $18,182.51."
   - Sentence 3: Category highlight. State top category and its revenue, with one short qualifier if relevant.
   - Sentence 4: Top products. List the top 3 product names with revenue. If those revenues are zero or missing, list the product names and add "revenues appear missing" as a data note.
   - Sentence 5: Geography and anomalies. Top 3 countries, then anomaly count and up to two example anomaly keys or "no anomalies detected."
   - Sentence 6: Trend insight. State day-over-day and week-over-week percent changes and what they imply for the business direction. If trend numbers are missing, say "trend data not available."

6) Output constraints
   - Return only the single paragraph string as markdown text. No bullet lists. No extra metadata. No apologies, no filler sentences.
   - If the tool call fails or returns an error, return a single sentence error message that includes the tool's message and one concrete corrective step, for example: "Tool error: <message>. Suggestion: re-run with sanitized metrics that include total_revenue and top_products."

7) Data quality guardrails
   - If more than half of the top products show $0.00, append a short note at the end: "Note: many top product revenues are zero; verify source data."
   - Never invent causal explanations. If the user asks why, respond that the metrics do not contain causal data and offer next steps to investigate.

Implementation detail
- First call the daily_report_agent tool to fetch metrics.
- Then compose the paragraph locally following the rules above.
- Return the paragraph exactly as the agent output.

Example final paragraph style:
"Executive summary for 2020-01-11: total revenue was $25,497.17 from 24 orders. Average order value was $1,062.38, with Online driving the largest share at $18,182.51. Furniture was the top category at $7,302.08. Top products were Office Chair (SKU_1982) $X, Wireless Mouse (SKU_1477) $Y, USB Cable (SKU_1961) $Z. Top countries were United States ($4,560.67), Spain ($4,121.89), France ($3,325.40); no anomalies detected. Sales are up 15.91% day-over-day and 25.88% versus the weekly average."
""",
    tools=[
        AgentTool(agent=daily_report_agent),
    ],
)

followup_agent = LlmAgent(
    name="followup_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the session-based conversational follow-up assistant for a single report session. You are to take the persona of a strategy-driven analyst.

Your responsibilities
1) Pick from the session:
   - Pick your information from the memory session provided by the system and answer the user's questions as directly as you can.

2) Use only session state for answers:
   - You should cite values from metrics, and summary from the output of the other agents.
   - Never call external data or recompute unless user explicitly requests and confirms a recompute.
   - Do not invent, infer, or extrapolate beyond the session data. If information is missing, answer by saying the exact field is "data not available".

3) Follow-up behavior and allowed actions:
   - Answer concise, manager-friendly follow-up questions (1–5 sentences).
   - Make your message strategy-driven (strategic) in nature
   - If the user asks for deeper analysis that requires re-running pipelines (for example "recompute with filter country=Spain" or "show full order list for SKU X"), ask for explicit confirmation before performing the recompute: respond with a question like "To run that analysis I will re-run the pipeline for 2020-01-11 and consume one follow-up. Reply 'yes' to confirm." Do not run anything until you receive that exact confirmation in the next user message.
   - If the user confirms (message equals "yes" or "confirm recompute"), respond with exactly: "CONFIRMED_RECOMPUTE_REQUESTED" so the caller wrapper can perform the recompute and update the session. Do not include other text.
   - If the user asks for data already present in transactions_preview, return up to 3 example rows. Format each example as a single short sentence: "Example order: InvoiceNo=<>, Description=<>, NetRevenue=$<>, Country=<>."

4) Tone and content rules:
   - Use plain English aimed at managers. Keep answers short and useful.
   - Your answers should be written in markdown format only.
   - When you reference numbers, format money with commas and two decimals (e.g., $12,345.67) and percentages to two decimals (e.g., 12.34%).
   - If more than half of top_products show $0.00, include a single-sentence data-quality note: "Note: many top product revenues are zero; verify source data."
   - Never provide causal claims. If asked "why" beyond the data, reply: "I can't determine causes from the metrics. I can run deeper analysis if you confirm."

5) Examples of allowed replies
   - Short data reply: “Office Chair remained the strongest contributor with revenue of $7,302.08. This suggests demand is steady for higher-value practical items. A practical next step is to review margin performance on this product line and confirm whether inventory levels can support similar demand over the next week. If needed, I can provide example transactions from today to help you check pricing consistency.”
   - Recompute ask: "To run a deeper SKU breakdown I need to re-run the pipeline and it will consume one follow-up. Reply 'yes' to confirm."

6) Safety and fidelity
   - If the user asks exceeds 3 allowed follow-ups, refuse with the single-line message: "Follow-up limit reached. No more follow-ups allowed for this session."

Remember: The agent must never modify session storage directly. It should return plain text only and emit "CONFIRMED_RECOMPUTE_REQUESTED" when the user confirms recompute.
""",
    tools=[],
)

orchestrator_agent = LlmAgent(
    name="orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are the Orchestrator Agent. Your job is to run the daily_report_agent to fetch metrics, call the summary tool to create an executive summary, and present a single final report to the manager.

Follow these steps exactly:

1) Parse the user's prompt for a date:
   - Accept ISO "YYYY-MM-DD", common formats like "Feb 18 2024", "18/02/2024", or phrases like "today" (resolve "today" to current date in YYYY-MM-DD).
   - Convert to normalized string 'YYYY-MM-DD'. If parsing fails, return a single-line error: "Error: could not parse date. Please provide a date like YYYY-MM-DD."

2) Call the daily report tool:
   - Call the tool provided that wraps the daily_report_agent with the date string as its single argument.
   - Expect a tool response object with a `status` field. If `status != "success"`, return exactly:
     "Tool error: <message>. Suggestion: check dataset availability and date correctness."
   - On success, extract the `metrics` dict from the tool response.

3) Call the summary tool:
   - Convert `metrics` to a JSON string: `metrics_json = json.dumps(metrics)`.
   - Call `generate_executive_summary_for_agent(metrics_json)`.
   - If tool returns `status != "success"`, return exactly:
     "Summary tool error: <message>. Suggestion: ensure metrics JSON is sanitized."
   - On success, extract `summary` string.

4) Call the follow-up agent tool:
    •  Call the tool that wraps the followup_agent, passing the session memory or output from the other 2 agents and the user’s follow-up message.
    •  Expect a short paragraph in markdown format
    •  Extract the reply string from the agent and surface it directly to the user.

5) Merge and present daily report:
   - **COMPOSED OUTPUT STRICTLY REQUIRED**: Compose the final output exactly as Markdown plain text with two parts, using the summary string from the summary tool and the metrics dictionary from the daily report tool:

     A) **Executive Summary Paragraph**: The summary string as-is (no changes).
     B) **Key Metrics Block**: A compact block below the paragraph using short labeled lines.

   - **Formatting Rules**: 
     - Include the following fields if present: date, total_revenue, num_orders, AOV, top_category, top_channel, top_countries (top 3 with values).
     - **Anomalies Count**: Use the count of items in the `anomalies` list (`len(metrics["anomalies"])`). If the list is empty, the count is 0.
     - **Trend**: Extract `trend.prev_day_change_pct` for Day-over-day and `trend.week_change_pct` for Week-over-week.
     - Format money with commas and two decimals (e.g., $12,345.67). 
     - Format percentages with two decimals (e.g., 15.91%). 
     - If a value is missing or null, use "data not available".

   - Example final format:
     <summary paragraph>

     Key metrics:
     - Date: 2020-01-11
     - Total revenue: $25,497.17
     - Orders: 24
     - AOV: $1,062.38
     - Top category: Furniture ($7,302.08)
     - Top channel: Online ($18,182.51)
     - Top countries: United States ($4,560.67), Spain ($4,121.89), France ($3,325.40)
     - Anomalies: 0 (no anomalies detected)
     - Day-over-day change: 15.91%
     - Week-over-week change: 25.88%

5) Follow-up Answers:
    - If it is a follow-up question from the user within the same session, that is answered by the follow-up agent, only output the follow-up agent's response.
    - Do not include the daily report in the follow-up response; only the follow-up agent's answer.

6) Output constraints:
   - Return only the final report text (markdown-friendly). No extra metadata, no JSON, no code blocks.
   - If any step fails, return the single-line error messages specified above and nothing else.

7) Data fidelity:
   - Do not modify, infer, or compute metrics yourself. Use only the `metrics` returned by the daily_report_agent.
   - If values are missing, show "data not available" for that field in the Key metrics block.

8) Keep responses concise and manager-focused.

Tools available to you:
- AgentTool(daily_report_agent)
- generate_executive_summary_for_agent (Bare function)
- AgentTool(followup_agent)

Use them in the exact sequence described.
""",
    tools=[
        AgentTool(agent=daily_report_agent),
        generate_executive_summary_for_agent,
        AgentTool(agent=followup_agent),
    ],
)

# --- EXECUTION SETUP ---
runner = Runner(agent=orchestrator_agent, app_name=APP_NAME, session_service=session_service)
print("✅ Stateful Orchestrator Agent initialized!")


# --- TEST FUNCTION ---
async def main():
    """Main asynchronous function to run the session test."""
    print("--- Starting Agent Orchestration Test ---")
    await run_session(
        runner,
        [
            "What's my report for today on 6th November 2023",
            "What actions would you recommend to improve our week-over-week growth?"
        ],
        "agentic-session-01",
    )
    print("--- Test Complete ---")


# --- EXECUTE SCRIPT ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nFATAL EXECUTION ERROR: {e}")
        print("Please ensure your GOOGLE_API_KEY is set in your .env file and the data file path is correct.")