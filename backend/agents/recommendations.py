import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from backend.analysis import analyze_vm_data

# Load environment variables
load_dotenv()

# Configure Google AI
GOOGLE_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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
    """
    
    if not GOOGLE_API_KEY:
        return {
            "error": "GOOGLE_AI_API_KEY not found in environment variables",
            "service": "Compute Engine",
            "insights": [],
            "recommendations": []
        }
    
    try:
        # First, analyze the data to get statistical insights
        analysis_results = analyze_vm_data(df)
        
        # Prepare a summary of the data for the AI model
        data_summary = prepare_data_summary(df, analysis_results)
        
        # Load agent context
        agent_context = load_agent_context()
        
        # Create the prompt for Gemini
        prompt = f"""
You are an expert cloud infrastructure analyst specializing in Google Cloud Platform VM optimization.

Use the following context and best practices to guide your analysis:

{agent_context}

---

Now analyze the following VM instance data and provide actionable insights and recommendations:

{data_summary}

Based on the context provided and the data above, please provide:
1. Key Insights: 3-5 important observations about the current VM usage patterns
2. Recommendations: 3-5 specific, actionable recommendations to optimize cost, performance, or reliability

Make sure to reference specific metrics from the data and apply the best practices from the context.

Format your response as JSON with this structure:
{{
    "insights": ["insight 1", "insight 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}}
"""
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse the response
        result = parse_gemini_response(response.text)
        
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


def parse_gemini_response(response_text: str) -> dict:
    """
    Parse the Gemini API response and extract structured data
    """
    try:
        # Remove markdown code blocks if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        result = json.loads(cleaned_text)
        
        return {
            'insights': result.get('insights', []),
            'recommendations': result.get('recommendations', [])
        }
    except json.JSONDecodeError:
        # Fallback: try to extract insights and recommendations manually
        return {
            'insights': ['Analysis completed but response format was unexpected'],
            'recommendations': ['Please review the raw analysis output']
        }