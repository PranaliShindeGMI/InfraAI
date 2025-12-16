def generate_vm_recommendations(df):
    # TODO: Replace with your ML + agent logic
    # Example output:
    return {
        "service": "Compute Engine",
        "insights": [
            "CPU usage stable over last 7 days.",
            "Network egress showed a spike yesterday."
        ],
        "recommendations": [
            "Monitor the spike in egress traffic.",
            "Potential opportunity for off-hour scheduling."
        ]
    }