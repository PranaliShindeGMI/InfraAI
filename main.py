# app.py
import streamlit as st
import requests
import streamlit.components.v1 as components
import json

st.set_page_config(layout="wide")

# ------- FETCH REAL DATA FROM BACKEND -------
BACKEND_URL = "http://localhost:8000/recommendations"

try:
    alerts = requests.get(BACKEND_URL).json()
except:
    st.warning("Alerts")

    alerts = alerts = [
    {
        "title": "Short CPU and Disk Spike Detected",
        "vm_instance": "vm-app-01",
        "impact_level": "Medium",
        "category": "Performance",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU, Disk Read, and Disk Write spiked together for a short duration, "
            "indicating a batch job or heavy workload running temporarily. "
            "This is normal if infrequent, but frequent spikes may cause latency "
            "for other workloads.\n\n"
            "Remediation:\n"
            "1. Schedule heavy jobs during off-peak hours.\n"
            "2. Enable autoscaling if spikes happen regularly."
        ),
    },
    {
        "title": "High Disk I/O With Low CPU Activity",
        "vm_instance": "vm-log-02",
        "impact_level": "Low",
        "category": "Storage",
        "cost_saved": "$0",
        "detailed_explanation": (
            "Disk I/O is high while CPU usage remains low, suggesting the instance "
            "is performing background file writes, log rotations, or inefficient I/O operations. "
            "This can increase disk latency and storage costs.\n\n"
            "Remediation:\n"
            "1. Check for redundant or overly frequent write operations.\n"
            "2. Move archived logs to a cheaper storage class like Nearline."
        ),
    },
    {
        "title": "High CPU Load With Low Disk Activity",
        "vm_instance": "vm-compute-03",
        "impact_level": "Medium",
        "category": "Performance",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU usage is elevated while Disk I/O remains low, indicating a compute-heavy "
            "workload such as compression, encryption, or data processing. "
            "This may lead to throttling if the CPU-to-memory ratio is poor.\n\n"
            "Remediation:\n"
            "1. Move to a machine type optimized for compute-heavy tasks.\n"
            "2. Evaluate CPU/memory ratio or consider custom machine types."
        ),
    },
    {
        "title": "Idle VM Instance Detected",
        "vm_instance": "vm-idle-04",
        "impact_level": "High",
        "category": "Cost",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU, Disk I/O, and Network activity have stayed consistently low for an extended period. "
            "This strongly suggests the VM is unused or underutilized, leading to unnecessary cost. "
            "An idle instance provides no operational value.\n\n"
            "Remediation:\n"
            "1. Stop or delete the VM if not required.\n"
            "2. Downsize the machine type to reduce ongoing cost."
        ),
    },
    {
        "title": "Sustained High Load Across All Metrics",
        "vm_instance": "vm-heavy-05",
        "impact_level": "High",
        "category": "Scaling",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU, Memory, and Disk activity remain consistently high, indicating an "
            "under-provisioned instance or a runaway process. Sustained pressure across resources "
            "can lead to failures or major slowdowns.\n\n"
            "Remediation:\n"
            "1. Upgrade to a larger instance or enable autoscaling.\n"
            "2. Investigate application behavior to rule out memory leaks."
        ),
    },
    {
        "title": "Rapid CPU Fluctuations With Constant Disk I/O",
        "vm_instance": "vm-app-06",
        "impact_level": "Medium",
        "category": "Reliability",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU usage fluctuates rapidly while Disk I/O remains constant, indicating unstable "
            "workload behavior or possible application misconfiguration. This instability can "
            "impact performance unpredictably.\n\n"
            "Remediation:\n"
            "1. Review autoscaling configuration for overly aggressive thresholds.\n"
            "2. Optimize application code to smooth CPU spikes."
        ),
    },
    {
        "title": "High Disk Write Activity With Low CPU",
        "vm_instance": "vm-backup-07",
        "impact_level": "Low",
        "category": "Storage",
        "cost_saved": "$0",
        "detailed_explanation": (
            "Disk Write Bytes are elevated while CPU usage remains low, indicating a backup, "
            "log rotation, or archival process running. This behavior is typically normal but "
            "may cause higher storage throughput cost.\n\n"
            "Remediation:\n"
            "1. Eliminate redundant backups or reduce backup frequency.\n"
            "2. Move backup targets to Nearline/Coldline storage."
        ),
    },
    {
        "title": "Oversized VM With Low Actual CPU Utilization",
        "vm_instance": "vm-large-08",
        "impact_level": "Medium",
        "category": "Cost",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU usage is high but CPU utilization (active cores used) is low—indicating the VM "
            "has too many vCPUs compared to the workload it is running. This means the machine "
            "is overprovisioned and wasting compute capacity.\n\n"
            "Remediation:\n"
            "1. Resize to a smaller VM type to reduce unused cores.\n"
            "2. Consider committed-use discounts for predictable workloads."
        ),
    },
    {
        "title": "Spike in Disk Reads With Steady CPU",
        "vm_instance": "vm-db-09",
        "impact_level": "Low",
        "category": "Performance",
        "cost_saved": "$0",
        "detailed_explanation": (
            "A sudden surge in Disk Read Bytes occurred while CPU remained steady, indicating "
            "a large dataset load or database read operation. This may be expected, but repeated "
            "spikes can cause IO latency.\n\n"
            "Remediation:\n"
            "1. Investigate query patterns and caching strategy.\n"
            "2. Add read replicas if load patterns increase."
        ),
    },
    {
        "title": "High Network Throughput With Low Disk Usage",
        "vm_instance": "vm-api-10",
        "impact_level": "High",
        "category": "Network",
        "cost_saved": "$0",
        "detailed_explanation": (
            "CPU utilization is moderately high while Disk Read and Write remain low, but "
            "Network egress is elevated. This indicates a heavy network-bound workload such "
            "as an API server or data streaming service.\n\n"
            "Remediation:\n"
            "1. Optimize network egress using caching or compression.\n"
            "2. Consider using a regional load balancer to distribute traffic."
        ),
    },
]

# ------- SEARCH + FILTER -------
col_search, col_filter = st.columns([4, 1.3])

with col_search:
    search = st.text_input("Search Alerts", placeholder="Search by alert or VM...")

with col_filter:
    severity_filter = st.selectbox("Severity", ["All", "High", "Medium", "Low"])

# ------- FILTER LOGIC -------
filtered = []
for a in alerts:
    # if search.lower() in a["title"].lower() or search.lower() in a["vm_instance"].lower():
    if severity_filter == "All" or a["impact_level"] == severity_filter:
        filtered.append(a)

# ------- TAILWIND + HTML UI -------
html = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lucide@latest"></script>

<style>
.row-hover:hover {
    transform: translateY(-2px);
    transition: 0.15s ease;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    cursor: pointer;
}
.drawer {
    position: fixed;
    top: 0;
    right: -450px;
    width: 420px;
    height: 100vh;
    background: white;
    transition: right .35s ease;
    z-index: 100;
    padding: 24px;
    overflow-y: auto;
    border-left: 1px solid #eee;
}
.overlay {
    position: fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background: rgba(0,0,0,0.35);
    display:none;
}
.drawer.open { right:0; }
.overlay.active { display:block; }
</style>

</head>

<body class="bg-[#f5f6f8] text-gray-800">

<div class="mt-4">

  <div class="grid grid-cols-7 font-semibold text-gray-600 text-sm border-b pb-3">
    <div class="col-span-2">Alert</div>
    <div>Category</div>
    <div>Impact</div>
    <div>VM</div>
    <div>Cost Saved</div>
  </div>
"""

# ------- ROWS -------
for item in filtered:
    html += f"""
    <div onclick='openDrawer({json.dumps(item)})'
         class="grid grid-cols-7 py-5 border-b items-start bg-white rounded-lg row-hover">

        <div class="col-span-2">
            <div class="font-semibold text-[15px] flex items-center gap-2">
                <i data-lucide="bell"></i> VM Instance
            </div>
            <div class="text-gray-500 text-sm leading-tight w-[90%]">{item['detailed_explanation'][:85]}...</div>
        </div>

        <div class="flex items-center">
            <span class="px-3 py-1 text-xs bg-gray-100 rounded-full text-gray-700 shadow-sm">
                {item['category']}
            </span>
        </div>

        <div class="flex items-center">
            <span class="px-3 py-1 text-xs text-white rounded-full shadow
                {'bg-rose-400' if item['impact_level']=='High' else 'bg-amber-300 text-black'}">
                {item['impact_level']}
            </span>
        </div>

        <div class="flex items-center text-sm">
            <i data-lucide="server" class="w-4 h-4 mr-1 text-gray-500"></i>
            {item['vm_instance']}
        </div>

        <div class="flex items-center text-green-600 font-semibold">
            <i data-lucide="dollar-sign" class="w-4 h-4 mr-1"></i>
            {item['cost_saved']}
        </div>
    </div>
    """

# ------- DRAWER -------
html += """
<div id="overlay" class="overlay"></div>

<div id="drawer" class="drawer">
    <h2 class="text-xl font-semibold flex items-center gap-2" id="drawer_title"></h2>
    <p class="mt-2 text-gray-600" id="drawer_desc"></p>

    <div class="mt-6">
        <h3 class="font-semibold text-gray-800 text-md">Details</h3>
        <p class="text-gray-700 mt-2 leading-relaxed" id="drawer_details"></p>
    </div>

    <button onclick="closeDrawer()"
            class="mt-6 bg-gray-200 px-4 py-2 rounded-md text-sm shadow">
        Close
    </button>
</div>

<script>
lucide.createIcons();

function openDrawer(data) {
    document.getElementById("drawer_title").innerText = data.title;
    document.getElementById("drawer_desc").innerText = "VM: " + data.vm_instance;
    document.getElementById("drawer_details").innerText = data.detailed_explanation;

    document.getElementById("drawer").classList.add("open");
    document.getElementById("overlay").classList.add("active");
}

function closeDrawer() {
    document.getElementById("drawer").classList.remove("open");
    document.getElementById("overlay").classList.remove("active");
}
</script>

</body>
</html>
"""

components.html(html, height=1100, scrolling=True)
