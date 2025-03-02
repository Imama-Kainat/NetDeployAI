import requests
import time
import folium
import json
import gradio as gr
import numpy as np
import os
from folium.plugins import MarkerCluster

# Hugging Face Inference API Key
HUGGINGFACE_API_KEY = "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"

def query_hf_model(prompt, max_retries=3, wait_time=10):
    """Queries the Falcon-7B-Instruct model from Hugging Face with retry logic."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}

    for attempt in range(max_retries):
        response = requests.post(
            "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            return response.json()[0]['generated_text']
        elif response.status_code == 503:
            return f"⚠ Model is busy, please try again later. (Attempt {attempt + 1}/{max_retries})"
            time.sleep(wait_time)
        else:
            return f"⚠ Error: {response.json()}"

    return "❌ Model is unavailable after multiple attempts."

def format_ai_response(response_text):
    """Formats the AI-generated response with line breaks for better readability."""
    try:
        response_json = json.loads(response_text)
        return json.dumps(response_json, indent=4)
    except json.JSONDecodeError:
        return response_text.replace("\\n", "\n").replace("\n    ", "\n")

def generate_map_html(lat, lon, coverage_area):
    """Generates a folium map with the deployment location marked and returns HTML."""
    network_map = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], popup="Deployment Location", icon=folium.Icon(color="blue")).add_to(network_map)
    folium.Circle(
        location=[lat, lon],
        radius=coverage_area * 1000,
        color="#3F4F44",
        fill=True,
        fill_opacity=0.2
    ).add_to(network_map)

    # Save map to HTML file
    map_html = network_map._repr_html_()
    return map_html

def parse_coordinates(location):
    """Parses latitude and longitude from user input."""
    try:
        location = location.replace("°", "").replace("N", "").replace("E", "").strip()
        lat, lon = map(float, location.split(','))
        return lat, lon
    except ValueError:
        return None, None

def get_network_suggestion(location, terrain, budget, special_conditions, estimated_users, coverage_area, technology_preference):
    """Generates network recommendations based on user inputs."""
    try:
        coverage_area_float = float(coverage_area)
    except ValueError:
        return "⚠ Invalid coverage area. Please enter a numeric value.", None

    lat, lon = parse_coordinates(location)

    if not lat or not lon:
        return "⚠ Invalid coordinate format. Please enter in 'lat, lon' format.", None

    prompt = f"""
    A user wants to deploy a network with the following requirements:
    - Location: {location}
    - Terrain: {terrain}
    - Budget: {budget}
    - Special Conditions: {special_conditions}
    - Estimated Users: {estimated_users}
    - Coverage Area: {coverage_area_float} km²
    - Technology Preference: {technology_preference}

    Provide a detailed deployment recommendation including:
    1. Recommended Network Topology
    2. Infrastructure Requirements
    3. Deployment Steps
    4. Cost Estimation & Budget Justification
    5. Scalability & Future Upgrades
    6. Potential Challenges & Mitigation Strategies
    """

    response = query_hf_model(prompt)

    if response:
        formatted_response = format_ai_response(response)
        map_html = generate_map_html(lat, lon, coverage_area_float)
        return formatted_response, map_html
    else:
        return "⚠ No response received.", None

def compare_network(model_suggestion, location, terrain, budget, special_conditions, estimated_users, coverage_area, technology_preference):
    """Compares the model's network deployment strategy with the user's suggestion."""
    try:
        coverage_area_float = float(coverage_area)
    except ValueError:
        return "⚠ Invalid coverage area. Please enter a numeric value."

    user_suggestion = f"""
    A user wants to deploy a network with the following requirements:
    - Location: {location}
    - Terrain: {terrain}
    - Budget: {budget}
    - Special Conditions: {special_conditions}
    - Estimated Users: {estimated_users}
    - Coverage Area: {coverage_area_float} km²
    - Technology Preference: {technology_preference}
    """

    comparison_prompt = f"""
    Compare the following two network deployment strategies based on efficiency, cost, and feasibility.
    *Model's Suggestion:* {model_suggestion}
    *User's Suggestion:* {user_suggestion}
    Provide a final decision on which one is better and explain why in a structured format.
    """

    comparison_response = query_hf_model(comparison_prompt)

    if comparison_response:
        return format_ai_response(comparison_response)
    else:
        return "⚠ No response received for comparison."

# Custom CSS for the application
css = """
:root {
    --primary: #3F4F44;
    --secondary: #A27B5C;
    --background: #E0E0E0
}

body {
    background-color: var(--background);
}

.gradio-container {
    background-color: var(--background);
}

.gr-button {
    background-color: var(--secondary) !important;
    color: white !important;
}

.gr-button:hover {
    background-color: var(--primary) !important;
}

.gr-form {
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.gr-input, .gr-textarea {
    border: 1px solid var(--secondary) !important;
}

.gr-input:focus, .gr-textarea:focus {
    border: 2px solid var(--primary) !important;
}

h1, h2, h3 {
    color: var(--primary);
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: var(--primary);
}

/* Sidebar styling */
.sidebar {
    background-color: var(--primary);
    border-radius: 10px;
    padding: 15px;
    color: white;

    height: 100vh; /* Full height */

    overflow-y: auto; /* Enable vertical scrolling */
}

.sidebar .gr-button {
    background-color: transparent !important;
    color: white !important;
    border: none;
    text-align: left;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 5px;
    width: 100%;
    border-left: 4px solid transparent;
}

.sidebar .gr-button:hover {
    background-color: rgba(255,255,255,0.1) !important;
    border-left: 4px solid var(--secondary);
}

.selected-nav {
    background-color: var(--secondary) !important;
    border-left: 4px solid white !important;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
}

.app-title {
    color: var(--primary);
    font-size: 28px;
    margin-bottom: 5px;
}

.app-subtitle {
    color: var(--secondary);
    font-size: 18px;
    margin-bottom: 10px;
}

.team-info {
    font-style: italic;
    margin-top: 10px;
    color: var(--primary);
}

/* Content containers */
.content-block {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    border-left: 5px solid var(--primary);
}
"""

# HTML for home page
home_html = """
<div class="app-header">
    <h1 class="app-title"> NetDeploy AI - Smart Network Deployment Assistant</h1>
    <h3 class="app-subtitle">AI-powered network deployment planning tool</h3>
    <p class="team-info">Made by team Dark Sky:<br>Imama Kainat | Abdul Hanan | Muhammad Hamza</p>
</div>

<div class="content-block">
    <h2 style="color: #3F4F44;">Welcome to NetDeploy AI</h2>
    <p>This application helps telecom engineers, ISPs, and businesses plan network deployments efficiently using AI.</p>
    <p>Use the sidebar to navigate between different sections:</p>
    <ul>
        <li><strong>About Application</strong> - Learn about features and capabilities</li>
        <li><strong>Get Network Deployment</strong> - Generate an AI-powered deployment plan</li>
        <li><strong>Compare Networks</strong> - Compare AI suggestions with your own plans</li>
    </ul>
</div>

<div class="content-block">
    <h2 style="color: #3F4F44;">How It Works</h2>
    <p>NetDeploy AI will analyze your requirements and generate optimized network deployment plans.</p>
    <p>Simply input your location, terrain, budget, and other details to receive comprehensive recommendations.</p>
</div>

<div class="footer">NetDeploy AI - Smart Network Deployment Assistant © 2025</div>
"""

# HTML for about page
about_html = """
<div class="app-header">
    <h1 class="app-title">About NetDeploy AI</h1>
</div>

<div class="content-block">
    <h2 style="color: #3F4F44;">📌 What Does This Application Do?</h2>
    <p>NetDeploy AI is an <strong>AI-powered network deployment assistant</strong> that helps <strong>telecom engineers, ISPs, and businesses</strong> efficiently plan and deploy network infrastructure. It takes in <strong>location, terrain, budget, technology preferences, and coverage area</strong> and generates <strong>a complete deployment strategy</strong> using the Falcon-7B-Instruct model.</p>

    <h3 style="color: #A27B5C;">Key Features:</h3>
    <ul>
        <li>✅ <strong>Network Topology</strong> – Recommends an optimized network layout (cell towers, fiber routes, etc.)</li>
        <li>✅ <strong>Infrastructure Requirements</strong> – Suggests required equipment (antennas, routers, fiber lines)</li>
        <li>✅ <strong>Budget Analysis</strong> – Estimates deployment costs & suggests budget-friendly options</li>
        <li>✅ <strong>Scalability & Expansion</strong> – Plans for future upgrades (5G, fiber expansions, etc.)</li>
        <li>✅ <strong>Challenges & Solutions</strong> – Identifies deployment challenges (terrain, weather, etc.) and mitigations</li>
        <li>✅ <strong>Interactive Map</strong> – Visualizes the deployment area with coverage radius</li>
    </ul>
</div>

<div class="content-block">
    <h2 style="color: #3F4F44;">📌 Who Can Use It?</h2>
    <ul>
        <li>🔹 <strong>Telecom Engineers & ISPs</strong> – Plan new network deployments or upgrade existing ones</li>
        <li>🔹 <strong>Government & Smart City Planners</strong> – Deploy infrastructure for public connectivity</li>
        <li>🔹 <strong>Enterprises & Startups</strong> – Assess feasibility of private networks (corporate campuses, factories, etc.)</li>
        <li>🔹 <strong>Rural Connectivity Initiatives</strong> – Deploy networks in remote or underserved areas</li>
    </ul>
</div>



<div class="footer">NetDeploy AI - Smart Network Deployment Assistant © 2025</div>
"""

# Create Gradio Interface with sidebar navigation
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            # Sidebar navigation using a Column with a custom class
            with gr.Column(elem_classes="sidebar"):
                gr.Markdown("### 📋 Navigation")
                home_btn = gr.Button("🏠 Home", elem_classes="nav-btn")
                about_btn = gr.Button("About Application", elem_classes="nav-btn")
                deploy_btn = gr.Button(" Network Deployment", elem_classes="nav-btn")
                compare_btn = gr.Button("Compare Networks", elem_classes="nav-btn")

        with gr.Column(scale=4):
            # Main content area with different sections
            with gr.Group(elem_id="home-section") as home_section:
                gr.HTML(home_html)

            with gr.Group(visible=False, elem_id="about-section") as about_section:
                gr.HTML(about_html)

            with gr.Group(visible=False, elem_id="deploy-section") as deploy_section:
                gr.Markdown("## 📡 Get Network Deployment Suggestion")
                with gr.Row():
                    with gr.Column():
                        location = gr.Textbox(label="🌍 Location (City, Country, or Coordinates)", placeholder="e.g., 40.7128, -74.0060")
                        terrain = gr.Textbox(label="📍 Terrain Type", placeholder="e.g., urban, hilly, desert")
                        budget = gr.Textbox(label="💰 Budget", placeholder="e.g., $50,000 or 'low'/'high'")
                        special_conditions = gr.Textbox(label="⚠ Special Conditions", placeholder="e.g., extreme weather, limited power")

                    with gr.Column():
                        estimated_users = gr.Textbox(label="👥 Estimated Users", placeholder="e.g., 5000")
                        coverage_area = gr.Textbox(label="📡 Coverage Area (km²)", placeholder="e.g., 10")
                        technology_preference = gr.Textbox(label="🔧 Preferred Network Technology", placeholder="e.g., 4G, 5G, satellite")
                        submit_btn = gr.Button("Generate Deployment Plan", variant="primary")

                output_text = gr.Textbox(label="\n\n\n📡 Generated Network Deployment Documentation", lines=15)
                output_map = gr.HTML(label="📍 Network Coverage Map")

                submit_btn.click(
                    fn=get_network_suggestion,
                    inputs=[location, terrain, budget, special_conditions, estimated_users, coverage_area, technology_preference],
                    outputs=[output_text, output_map]
                )

            with gr.Group(visible=False, elem_id="compare-section") as compare_section:

                with gr.Row():
                    with gr.Column():

                        gr.Markdown("### 📊 Compare AI-Generated Plan with Your Own")
                        ai_suggestion = gr.Textbox(label="AI-Generated Plan (Copy from first tab)", lines=5, placeholder="Paste the AI-generated plan here")

                        gr.Markdown("### 🚀 Your Network Deployment Strategy")
                        user_location = gr.Textbox(label="🌍 Location", placeholder="e.g., 40.7128, -74.0060")
                        user_terrain = gr.Textbox(label="📍 Terrain Type", placeholder="e.g., urban, hilly, desert")
                        user_budget = gr.Textbox(label="💰 Budget", placeholder="e.g., $50,000 or 'low'/'high'")
                        user_special_conditions = gr.Textbox(label="⚠ Special Conditions", placeholder="e.g., extreme weather, limited power")
                        user_estimated_users = gr.Textbox(label="👥 Estimated Users", placeholder="e.g., 5000")
                        user_coverage_area = gr.Textbox(label="📡 Coverage Area (km²)", placeholder="e.g., 10")
                        user_technology_preference = gr.Textbox(label="🔧 Preferred Network Technology", placeholder="e.g., 4G, 5G, satellite")

                        compare_submit_btn = gr.Button("Compare Networks", variant="primary")

                comparison_output = gr.Textbox(label="📊 Comparison Result", lines=15)

                compare_submit_btn.click(
                    fn=compare_network,
                    inputs=[ai_suggestion, user_location, user_terrain, user_budget, user_special_conditions,
                            user_estimated_users, user_coverage_area, user_technology_preference],
                    outputs=comparison_output
                )

    # Navigation logic
    all_sections = [home_section, about_section, deploy_section, compare_section]

    def nav_to(section_index):
        return [gr.update(visible=(i == section_index)) for i in range(len(all_sections))]

    home_btn.click(lambda: nav_to(0), inputs=None, outputs=all_sections)
    about_btn.click(lambda: nav_to(1), inputs=None, outputs=all_sections)
    deploy_btn.click(lambda: nav_to(2), inputs=None, outputs=all_sections)
    compare_btn.click(lambda: nav_to(3), inputs=None, outputs=all_sections)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
