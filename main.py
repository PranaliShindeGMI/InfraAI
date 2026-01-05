# app.py
import streamlit as st
import requests
import streamlit.components.v1 as components
import json
from utilities import alertMockData

st.set_page_config(layout="wide")

# ------- FETCH REAL DATA FROM BACKEND -------
BACKEND_URL = "http://localhost:8000/vm/recommendations"

# Specific alert data
specific_alert = {
    "title": "VM 5551001: Persistent CPU Underutilization Detected",
    "vm_instance": "5551001",
    "impact_level": "High",
    "category": "Cost",
    "detailed_explanation": "Instance 5551001 consistently shows very low CPU utilization, averaging around 7.0-7.5% over the forecast period (October 20-24, 2025). This is significantly below the optimal 40-70% range and the <30% underutilization threshold, indicating the instance is heavily oversized for its workload. Right-sizing this VM to a smaller machine type, potentially an E2 series, will lead to substantial cost savings without impacting performance. Consider using a custom machine type to match precise resource needs.",
    "cost": 6,
    "cost_explanation": "Current daily cost forecast is ~$0.40. Monthly cost is ~$12.00. Assuming a 50% reduction in cost by right-sizing to a smaller machine type (e.g., E2-small or custom machine type) suitable for consistently low CPU usage, estimated monthly savings are $12.00 * 0.50 = $6.00."
}

try:
    alerts = requests.get(BACKEND_URL).json()
    # Add the specific alert to the beginning of the list
    alerts.insert(0, specific_alert)
    # st.warning("Alerts: Fetched from backend.")
except:
    # st.warning("Alerts: Using mock data.")
    alerts = [specific_alert] + alertMockData.alerts
    

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

  <div class="grid grid-cols-8 font-semibold text-gray-600 text-sm border-b pb-3">
    <div class="col-span-3">Alert</div>
    <div>Category</div>
    <div>Impact</div>
    <div>VM</div>
    <div>Monthly Savings</div>
  </div>
"""

# ------- ROWS -------
for item in filtered:
    cost_display = f"${item.get('cost', 0)}" if item.get('cost') else "N/A"
    html += f"""
    <div onclick='openDrawer({json.dumps(item)})'
         class="grid grid-cols-8 py-5 border-b items-start bg-white rounded-lg row-hover">

        <div class="col-span-3">
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
                {'bg-rose-400' if item['impact_level']=='High' else 'bg-green-500' if item['impact_level']=='Low' else 'bg-amber-300 text-black'}">
                {item['impact_level']}
            </span>
        </div>

        <div class="flex items-center text-sm">
            <i data-lucide="server" class="w-4 h-4 mr-1 text-gray-500"></i>
            {item['vm_instance']}
        </div>

        <div class="flex items-center">
            <span class="text-sm font-semibold text-green-600">{cost_display}</span>
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

    <div class="mt-6" id="cost_section" style="display:none;">
        <h3 class="font-semibold text-gray-800 text-md">Cost Impact</h3>
        <div class="mt-2 bg-green-50 border border-green-200 rounded-lg p-4">
            <p class="text-2xl font-bold text-green-600" id="drawer_cost"></p>
            <p class="text-sm text-gray-600 mt-2 leading-relaxed" id="drawer_cost_explanation"></p>
        </div>
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

    // Show cost section if cost data is available
    if (data.cost !== undefined && data.cost_explanation) {
        document.getElementById("cost_section").style.display = "block";
        document.getElementById("drawer_cost").innerText = "Potential Savings: $" + data.cost + "/month";
        document.getElementById("drawer_cost_explanation").innerText = data.cost_explanation;
    } else {
        document.getElementById("cost_section").style.display = "none";
    }

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
