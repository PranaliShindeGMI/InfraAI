import os
import json
import asyncio
from dotenv import load_dotenv
import pandas as pd

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from backend.analysis import analyze_vm_data
from backend.vm_recommender.agent import root_agent as vm_recommender_agent

# Load environment variables
load_dotenv()

# Load agent context
CONTEXT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'agent_context.txt')
 
def load_agent_context() -> str:
    """
    Load the agent context from the context file
    """
    try:
        if os.path.exists(CONTEXT_FILE_PATH):
            with open(CONTEXT_FILE_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "No additional context available."
    except Exception as e:
        print(f"Warning: Could not load agent context: {e}")
        return "No additional context available."
 
def generate_vm_recommendations(df):
    """
    Generate AI-powered recommendations based on VM data analysis
    using Google's Gemini model
   
    Returns a list of alert objects in the format:
    [
        {
            "title": "Short description",
            "vm_instance": "instance name",
            "impact_level": "Low/Medium/High",
            "category": "Performance/Storage/Cost/etc",
            "detailed_explanation": "Long description with remediation steps"
        },
        ...
    ]
    """
   
    if not GOOGLE_API_KEY:
        return [{
            "title": "Configuration Error",
            "vm_instance": "N/A",
            "impact_level": "High",
            "category": "Configuration",
            "detailed_explanation": "GOOGLE_AI_API_KEY not found in environment variables. Please configure the API key to generate recommendations."
        }]
   
    try:
        # First, analyze the data to get statistical insights
        analysis_results = analyze_vm_data(df)
       
        # Prepare a summary of the data for the AI model
        data_summary = prepare_data_summary(df, analysis_results)
       
        # Load agent context
        agent_context = load_agent_context()
       
        forecast_csv_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'VM', 'data', 'forecast', 'vm_forecasts.csv')
        forecasted_values_df = pd.read_csv(forecast_csv_path)
        forecast_summary = forecasted_values_df.to_string()
        print("Forecast Summary Loaded for Recommendations.",forecasted_values_df.shape)
        # Create the prompt for Gemini
        prompt = f"""
Analyze the following VM instance data and generate infrastructure alerts.
 
DATA SUMMARY:
{data_summary}
 
FORECASTED VALUES (next 5 days):
{forecast_summary}

TASK: Generate 5-8 VM alerts based on the data above indicating potential issues or optimization opportunities. Do not include any alerts involving shortcomings of data.
 
Examples of alert:
"title": "High Network Throughput With Low Disk Usage",
        "vm_instance": "vm-api-10",
        "impact_level": "High",
        "category": "Network",
        "detailed_explanation": (
            "CPU utilization is moderately high while Disk Read and Write remain low, but "
            "Network egress is elevated. This indicates a heavy network-bound workload such "
            "as an API server or data streaming service.\n\n"
            "Remediation:\n"
            "1. Optimize network egress using caching or compression.\n"
            "2. Consider using a regional load balancer to distribute traffic."
        ),
"title": "Spike in Disk Reads With Steady CPU",
        "vm_instance": "vm-db-09",
        "impact_level": "Low",
        "category": "Performance",
        "detailed_explanation": (
            "A sudden surge in Disk Read Bytes occurred while CPU remained steady, indicating "
            "a large dataset load or database read operation. This may be expected, but repeated "
            "spikes can cause IO latency.\n\n"
            "Remediation:\n"
            "1. Investigate query patterns and caching strategy.\n"
            "2. Add read replicas if load patterns increase."
        ),        
 
CRITICAL: You MUST respond with ONLY a valid JSON array. Do not include any explanatory text, markdown, or commentary.
 
Required JSON format (respond with ONLY this structure):
[
    {{
        "title": "Short description (5-10 words)",
        "vm_instance": "instance-id-from-data",
        "impact_level": "Low|Medium|High",
        "category": "Performance|Cost|Storage|Network|Reliability|Scaling",
        "detailed_explanation": "Full explanation with metrics and remediation steps"
    }}
]
 
Use these categories: Performance, Cost, Storage, Network, Reliability, Scaling
For vm_instance, use actual VM IDs from the data summary if available, otherwise use descriptive names like "vm-high-cpu-01"
"""
       
        # Initialize Gemini model with JSON mode
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                "response_mime_type": "application/json"
            }
        )
       
        # Generate response
        response = model.generate_content(prompt)
       
        # Parse the response
        alerts = parse_gemini_alerts_response(response.text)
        return alerts
       
        data_summary = prepare_data_summary(df, analysis_results)
        
        # Call vm_recommender_agent via ADK Runner, similar to agent_run.py
        async def _call_agent(summary: str) -> str:
            session_service = InMemorySessionService()
            await session_service.create_session(
                app_name="vm_recommender_app",
                user_id="backend_user",
                session_id="vm_recommendations_session",
                state={"data_summary": summary},
            )

            runner = Runner(
                agent=vm_recommender_agent,
                app_name="vm_recommender_app",
                session_service=session_service,
            )

            content = types.Content(role="user", parts=[types.Part(text=summary)])

            final_text = None
            async for event in runner.run_async(
                user_id="backend_user",
                session_id="vm_recommendations_session",
                new_message=content,
            ):
                if event.is_final_response():
                    parts = (
                        event.content.parts
                        if event.content and event.content.parts
                        else []
                    )
                    final_text = "".join(p.text or "" for p in parts)
                    break

            return final_text or "{}"

        # Run the async agent call synchronously
        agent_response_text = asyncio.run(_call_agent(data_summary))

        # Parse the response (expects the JSON structure defined in the agent)
        result = parse_gemini_response(agent_response_text)
        
        # Add service information
        result['service'] = 'Compute Engine'
        result['model_used'] = 'gemini-2.5-flash'
        
        return result
        
    except Exception as e:
        return {
            "error": f"Failed to generate recommendations: {str(e)}",
            "service": "Compute Engine",
            "insights": ["Error occurred during analysis"],
            "recommendations": ["Please check API key and try again"]
        }


def prepare_data_summary(df: pd.DataFrame, analysis_results: dict) -> str:
    """
    Prepare a concise summary of VM data for the AI model
    """
    summary_parts = []
   
    # Basic statistics
    summary_parts.append(f"Total VM Instances: {len(df)}")
   
    # Key metrics from descriptive stats
    if 'descriptive_stats' in analysis_results:
        stats = analysis_results['descriptive_stats']
       
        if 'cpu_utilization_mean' in stats:
            cpu_stats = stats['cpu_utilization_mean']
            summary_parts.append(f"\nCPU Utilization:")
            summary_parts.append(f"  - Average: {cpu_stats['mean']:.2f}%")
            summary_parts.append(f"  - Range: {cpu_stats['min']:.2f}% - {cpu_stats['max']:.2f}%")
       
        if 'memory_used_gb_mean' in stats:
            mem_stats = stats['memory_used_gb_mean']
            summary_parts.append(f"\nMemory Usage:")
            summary_parts.append(f"  - Average: {mem_stats['mean']:.2f} GB")
            summary_parts.append(f"  - Range: {mem_stats['min']:.2f} - {mem_stats['max']:.2f} GB")
       
        if 'cost_usd_sum' in stats:
            cost_stats = stats['cost_usd_sum']
            summary_parts.append(f"\nCost:")
            summary_parts.append(f"  - Total: ${cost_stats['mean'] * len(df):.2f}")
            summary_parts.append(f"  - Average per VM: ${cost_stats['mean']:.2f}")
   
    # Utilization insights
    if 'utilization_insights' in analysis_results:
        insights = analysis_results['utilization_insights']
        summary_parts.append(f"\nResource Utilization:")
        summary_parts.append(f"  - Underutilized VMs: {insights.get('underutilized_vms', 0)}")
        summary_parts.append(f"  - Overutilized VMs: {insights.get('overutilized_vms', 0)}")
        summary_parts.append(f"  - Well-utilized VMs: {insights.get('well_utilized_vms', 0)}")
   
    # Anomalies
    if 'anomalies' in analysis_results:
        anomalies = analysis_results['anomalies']
        if anomalies.get('total_anomalies', 0) > 0:
            summary_parts.append(f"\nAnomalies Detected: {anomalies['total_anomalies']}")
            if 'anomaly_types' in anomalies:
                summary_parts.append("  Types: " + ", ".join(anomalies['anomaly_types']))
   
    # Top instances by cost (if available)
    if 'cost_usd_sum' in df.columns and 'vm_instance_id' in df.columns:
        top_costly = df.nlargest(3, 'cost_usd_sum')[['vm_instance_id', 'cost_usd_sum']]
        summary_parts.append(f"\nTop 3 Costly VMs:")
        for _, row in top_costly.iterrows():
            summary_parts.append(f"  - {row['vm_instance_id']}: ${row['cost_usd_sum']:.2f}")
   
    return "\n".join(summary_parts)
 
 
def parse_gemini_alerts_response(response_text: str) -> list:
    """
    Parse the Gemini API response and extract alert objects
   
    Returns a list of alert dictionaries with keys:
    - title
    - vm_instance
    - impact_level
    - category
    - detailed_explanation
    """
    try:
        # Remove markdown code blocks if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
       
        # Log the cleaned text for debugging
        print(f"DEBUG: Cleaned response text (first 200 chars): {cleaned_text[:200]}")
       
        # Parse JSON
        alerts = json.loads(cleaned_text)
       
        # Validate and ensure correct format
        if isinstance(alerts, list):
            validated_alerts = []
            for alert in alerts:
                if isinstance(alert, dict):
                    validated_alerts.append({
                        'title': alert.get('title', 'Unknown Issue'),
                        'vm_instance': alert.get('vm_instance', 'N/A'),
                        'impact_level': alert.get('impact_level', 'Medium'),
                        'category': alert.get('category', 'Performance'),
                        'detailed_explanation': alert.get('detailed_explanation', 'No details available.')
                    })
            return validated_alerts if validated_alerts else _get_fallback_alert()
        else:
            print(f"DEBUG: Response is not a list, it's a {type(alerts)}")
            return _get_fallback_alert()
           
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decode error: {str(e)}")
        print(f"DEBUG: Raw response (first 500 chars): {response_text[:500]}")
        return _get_fallback_alert()
    except Exception as e:
        print(f"DEBUG: Unexpected error in parsing: {str(e)}")
        return _get_fallback_alert()
 
 
def _get_fallback_alert() -> list:
    """
    Return a fallback alert when parsing fails
    """
    return [{
        'title': 'Analysis Completed - Format Issue',
        'vm_instance': 'N/A',
        'impact_level': 'Low',
        'category': 'System',
        'detailed_explanation': 'Analysis completed but the response format was unexpected. Please review the configuration.\n\nRemediation:\n1. Check the Gemini API response format\n2. Verify the prompt template\n3. Review the error logs'
    }]