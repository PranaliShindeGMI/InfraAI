# main.py
import os
from agent import root_agent # Import the agent defined in agent.py

# --- Step 1: (Already covered in Environment Setup) ---
# Ensure Google Cloud authentication is set up.
# For example, by running `gcloud auth application-default login`
# or setting the GOOGLE_APPLICATION_CREDENTIALS environment variable.

# --- Step 2: Prepare the input data for the agent ---
# The agent's instruction expects a 'data_summary' string.
# This summary should contain details about the VM instances' usage.
# In a real-world scenario, this data would typically be fetched from
# your monitoring systems (e.g., Cloud Monitoring, custom scripts).
# For this example, we'll use a sample string.
data_summary = """
VM Instance Usage Data Summary (Last 30 Days):

Instance ID: vm-prod-001
  - Machine Type: n1-standard-4 (4 vCPU, 16GB RAM)
  - Average CPU Utilization: 65% (peaks at 90% during business hours, drops to 15% overnight)
  - Average Memory Utilization: 70% (stable)
  - Disk I/O: Moderate (500 IOPS average)
  - Network Egress: High (100 GB/day)
  - Cost: $200/month

Instance ID: vm-dev-002
  - Machine Type: e2-medium (2 vCPU, 4GB RAM)
  - Average CPU Utilization: 10% (rarely exceeds 20%)
  - Average Memory Utilization: 25% (stable)
  - Disk I/O: Very Low
  - Network Egress: Low
  - Cost: $30/month
  - Note: This instance is only used during weekdays, 9 AM - 5 PM.

Instance ID: vm-prod-003
  - Machine Type: n2-highmem-8 (8 vCPU, 64GB RAM)
  - Average CPU Utilization: 85% (consistently high, often hitting 95-100%)
  - Average Memory Utilization: 90% (consistently high, with occasional OOM errors)
  - Disk I/O: High (1500 IOPS average)
  - Network Egress: Moderate
  - Cost: $450/month

Instance ID: vm-test-004
  - Machine Type: e2-small (2 vCPU, 2GB RAM)
  - Average CPU Utilization: 5% (idle most of the time)
  - Average Memory Utilization: 10% (idle most of the time)
  - Disk I/O: Negligible
  - Network Egress: Negligible
  - Cost: $20/month
  - Note: This instance is always on but only used for ad-hoc testing.

Instance ID: vm-batch-005
  - Machine Type: c2-standard-8 (8 vCPU, 32GB RAM)
  - Average CPU Utilization: 40% (spikes to 80% for 4 hours daily during batch jobs, then drops to 5%)
  - Average Memory Utilization: 30% (stable)
  - Disk I/O: High during batch jobs, low otherwise
  - Network Egress: Low
  - Cost: $350/month
  - Note: Batch jobs run daily from 1 AM to 5 AM UTC.
"""

# The input to the agent's run method must be a dictionary
agent_input = {"data_summary": data_summary}

# --- Step 3: Call the agent to get a response ---
print("Calling the VM Recommender Agent...")
try:
    # The run() method returns an object conforming to the output_schema
    # In this case, it will be an instance of RecommendationResponseSchema.
    response_object = root_agent.run(agent_input)

    # --- Step 4: Process and display the response ---
    # Access the insights and recommendations from the response object.
    insights = response_object.insights
    recommendations = response_object.recommendations

    print("\n--- Agent Response ---")
    print("\nKey Insights:")
    for i, insight in enumerate(insights):
        print(f"{i+1}. {insight}")

    print("\nRecommendations:")
    for i, recommendation in enumerate(recommendations):
        print(f"{i+1}. {recommendation}")

except Exception as e:
    print(f"An error occurred while calling the agent: {e}")
    print("Please ensure your Google Cloud authentication is set up correctly, the ADK library is installed, and the Gemini model is accessible.")