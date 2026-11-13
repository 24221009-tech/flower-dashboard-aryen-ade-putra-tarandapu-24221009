"""streamlit_app.py: Flower Federated Learning Interactive Dashboard.

Simulates 1 Server (Flower SuperLink) & 3 SuperNode Clients with real-time visual interaction flow:
Broadcast -> Local Training -> Upload Weights -> FedAvg Aggregation.
Includes:
- Interactive HTML/CSS/SVG + JavaScript canvas with clickable nodes & detail modals (always active)
- Alive & continuous synchronization animation on COMPLETED status (two-way glowing particles)
- Step-by-Step execution (1 round per click) & Automatic multi-round execution
- "🎬 Putar Ulang Animasi Visual" button / Demo Loop Mode
- Dynamic Client Online/Offline connection simulator with adaptive FedAvg aggregation
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure local package import works
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from mlxexample.task import (
    load_model,
    get_weights,
    set_weights,
    generate_partition_data,
    train,
    test,
    get_framework,
    HAS_MLX,
)
import torch
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Flower Dashboard | 1 Server & 3 Clients Simulation",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Flower Dashboard Modern Theme */
    .flower-navbar {
        background: #0f172a;
        color: white;
        padding: 14px 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .flower-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .flower-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        display: inline-block;
    }
    .terminal-box {
        background: #0f172a;
        color: #38bdf8;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.82rem;
        padding: 14px;
        border-radius: 10px;
        height: 190px;
        overflow-y: auto;
        line-height: 1.45;
        border: 1px solid #1e293b;
    }
    .terminal-line { margin-bottom: 4px; }
    .term-ts { color: #64748b; }
    .term-srv { color: #f43f5e; font-weight: bold; }
    .term-cli { color: #10b981; font-weight: bold; }
    .term-fed { color: #a855f7; font-weight: bold; }
    .term-sync { color: #06b6d4; font-weight: bold; }
    .term-warn { color: #f59e0b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "current_round" not in st.session_state:
    st.session_state.current_round = 0
if "current_phase" not in st.session_state:
    st.session_state.current_phase = "READY"
if "global_model" not in st.session_state:
    st.session_state.global_model = load_model()
if "global_weights" not in st.session_state:
    st.session_state.global_weights = get_weights(st.session_state.global_model)
if "partitions_cache" not in st.session_state:
    st.session_state.partitions_cache = generate_partition_data(num_partitions=3, iid=True)
if "logs" not in st.session_state:
    st.session_state.logs = [
        f"[{time.strftime('%H:%M:%S')}] [Flower SuperLink] Inisialisasi ServerApp dan SuperLink selesai.",
        f"[{time.strftime('%H:%M:%S')}] [Flower SuperLink] 3 SuperNode Clients siap terhubung.",
        f"[{time.strftime('%H:%M:%S')}] [Flower ServerApp] Strategi FedAvg diaktifkan.",
        f"[{time.strftime('%H:%M:%S')}] [Flower Ready] Klik '▶ Mulai Otomatis' atau '⏭️ Jalankan 1 Ronde' untuk memulai."
    ]
if "client_states" not in st.session_state:
    st.session_state.client_states = {
        0: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": True},
        1: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": True},
        2: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": True},
    }
if "server_state" not in st.session_state:
    st.session_state.server_state = {"status": "Standby", "loss": 0.0, "acc": 0.0}

# -----------------------------------------------------------------------------
# Top Navigation Header (Flower Dashboard Brand)
# -----------------------------------------------------------------------------
framework_name = get_framework()
st.markdown(f"""
<div class="flower-navbar">
    <div class="flower-title">
        <span>🌸</span>
        <span>FLOWER FEDERATED DASHBOARD</span>
        <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8; margin-left: 8px;">| 1 Server (SuperLink) & 3 Clients (SuperNodes)</span>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
        <span class="flower-status-pill"><span class="pulse-dot"></span> FEDERATION ACTIVE</span>
        <span style="font-size: 0.8rem; background: #1e293b; color: #cbd5e1; padding: 4px 12px; border-radius: 14px; border: 1px solid #334155;">
            Backend: {framework_name}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://raw.githubusercontent.com/adap/flower/main/src/py/flwr/server/app/flwr-logo.svg", width=170)
st.sidebar.title("🎛️ Pengaturan Federasi")

st.sidebar.markdown("### Konfigurasi SuperLink")
num_rounds = st.sidebar.slider("Target Jumlah Ronde", min_value=1, max_value=10, value=3, step=1)
st.sidebar.caption("Simulasi akan berhenti otomatis ketika mencapai target ronde ini.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Konfigurasi SuperNodes")
local_epochs = st.sidebar.slider("Local Epochs per Klien", min_value=1, max_value=5, value=1, step=1)
learning_rate = st.sidebar.slider("Learning Rate (SGD)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
data_distribution = st.sidebar.radio(
    "Distribusi Data Partisi",
    ["IID (Seimbang across 10 digit)", "Non-IID (Partisi spesifik tiap client)"],
    index=0
)
is_iid = "IID" in data_distribution

st.sidebar.markdown("---")
st.sidebar.markdown("### Kecepatan Animasi")
anim_delay = st.sidebar.slider("Delay Animasi per Fase (detik)", min_value=0.2, max_value=2.0, value=0.6, step=0.1)

# -----------------------------------------------------------------------------
# Main Control Toolbar & Client Connection Toggles
# -----------------------------------------------------------------------------
st.markdown("### 🎛️ Kontrol Simulasi & Status Koneksi Klien")

with st.container():
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.5, 1.5, 1.4, 0.9])
    btn_auto = col_btn1.button("▶ Mulai Otomatis", type="primary", use_container_width=True)
    btn_step = col_btn2.button("⏭️ Jalankan 1 Ronde", use_container_width=True)
    btn_demo = col_btn3.button("🎬 Putar Ulang Animasi", use_container_width=True)
    btn_reset = col_btn4.button("🔄 Reset", use_container_width=True)

    # Interactive Client Connection Toggles
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#334155; margin-top:8px; margin-bottom:4px;'>📡 Saklar Koneksi Klien (Simulasikan Client Offline/Online):</div>", unsafe_allow_html=True)
    c_tog1, c_tog2, c_tog3 = st.columns(3)
    
    c1_online = c_tog1.toggle("Client 1 (SuperNode #1)", value=st.session_state.client_states[0].get("online", True), key="toggle_client_1")
    c2_online = c_tog2.toggle("Client 2 (SuperNode #2)", value=st.session_state.client_states[1].get("online", True), key="toggle_client_2")
    c3_online = c_tog3.toggle("Client 3 (SuperNode #3)", value=st.session_state.client_states[2].get("online", True), key="toggle_client_3")

    st.session_state.client_states[0]["online"] = c1_online
    st.session_state.client_states[1]["online"] = c2_online
    st.session_state.client_states[2]["online"] = c3_online

    active_clients = [c1_online, c2_online, c3_online]
    num_online = sum(active_clients)

if btn_reset:
    st.session_state.history = []
    st.session_state.current_round = 0
    st.session_state.current_phase = "READY"
    st.session_state.global_model = load_model()
    st.session_state.global_weights = get_weights(st.session_state.global_model)
    st.session_state.partitions_cache = generate_partition_data(num_partitions=3, iid=is_iid)
    st.session_state.logs = [
        f"[{time.strftime('%H:%M:%S')}] [Flower Reset] Riwayat simulasi telah di-reset ke kondisi awal.",
        f"[{time.strftime('%H:%M:%S')}] [Flower Ready] Siap untuk memulai simulasi federasi..."
    ]
    st.session_state.server_state = {"status": "Standby", "loss": 0.0, "acc": 0.0}
    st.session_state.client_states = {
        0: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": c1_online},
        1: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": c2_online},
        2: {"status": "Standby", "loss": 0.0, "acc": 0.0, "online": c3_online},
    }
    st.rerun()

# -----------------------------------------------------------------------------
# Function: Generate Dynamic Interactive HTML/CSS/SVG + JS Topology Diagram
# -----------------------------------------------------------------------------
def build_interactive_topology_html(phase: str, server_state: dict, client_states: dict, current_round: int, total_rounds: int) -> str:
    """Build SVG + HTML + JavaScript visual diagram with clickable nodes, popup modals, and continuous lively animations."""
    
    phase_cfg = {
        "READY": ("#1D4ED8", "#EFF6FF", "#93C5FD", "Siap Memulai — Klik 'Mulai Otomatis' atau 'Jalankan 1 Ronde' di atas"),
        "BROADCAST": ("#1D4ED8", "#EFF6FF", "#93C5FD", f"📡 [Ronde {current_round}/{total_rounds}] FASE 1: Broadcast — Server mengirim Bobot Global (W_t) ke Klien Online (Garis Biru)"),
        "LOCAL_TRAIN": ("#047857", "#ECFDF5", "#6EE7B7", f"🏋️ [Ronde {current_round}/{total_rounds}] FASE 2: Local Training — SuperNode Klien aktif melatih model secara lokal (Glow Hijau)"),
        "UPLOAD_WEIGHTS": ("#C2410C", "#FFF7ED", "#FDBA74", f"⬆️ [Ronde {current_round}/{total_rounds}] FASE 3: Upload Weights — Klien aktif mengunggah parameter bobot ΔW ke Server (Garis Oranye)"),
        "FEDAVG_AGG": ("#7E22CE", "#FAF5FF", "#D8B4FE", f"🧮 [Ronde {current_round}/{total_rounds}] FASE 4: FedAvg Aggregation — Server mengagregasi bobot dari klien aktif (Glow Ungu-Emas)"),
        "COMPLETED": ("#065f46", "#ECFDF5", "#6EE7B7", f"✨ [Ronde {current_round}/{total_rounds}] Simulasi Selesai: Model Global Terdistribusi & Sinkron (Aliran Partikel Hijau-Biru Berkelanjutan)"),
    }
    badge_color, badge_bg, badge_border, phase_text = phase_cfg.get(phase, ("#1D4ED8", "#EFF6FF", "#93C5FD", phase))

    srv_acc = server_state.get("acc", 0.0)
    srv_loss = server_state.get("loss", 0.0)

    # Server styling & animation based on phase
    if phase == "FEDAVG_AGG":
        srv_stroke = "#a855f7"
        srv_fill = "#faf5ff"
        srv_glow = "filter: url(#glow-purple); animation: pulseServer 1.1s infinite alternate;"
        srv_badge_bg = "#f3e8ff"
        srv_badge_color = "#7e22ce"
        srv_badge_text = "🧮 Aggregating FedAvg Weights..."
    elif phase == "BROADCAST":
        srv_stroke = "#3b82f6"
        srv_fill = "#eff6ff"
        srv_glow = "filter: url(#glow-blue);"
        srv_badge_bg = "#dbeafe"
        srv_badge_color = "#1d4ed8"
        srv_badge_text = "📡 Menyiarkan Bobot Global (W_t)"
    elif phase == "COMPLETED":
        srv_stroke = "#10b981"
        srv_fill = "#f0fdf4"
        srv_glow = "filter: url(#glow-green); animation: pulseCompleted 1.6s infinite alternate;"
        srv_badge_bg = "#d1fae5"
        srv_badge_color = "#065f46"
        srv_badge_text = "✨ Model Global Terdistribusi & Sinkron"
    else:
        srv_stroke = "#FF007A"
        srv_fill = "#ffffff"
        srv_glow = ""
        srv_badge_bg = "#ffe4e6"
        srv_badge_color = "#e11d48"
        srv_badge_text = "Standby (SuperLink)"

    # Coordinates for lines
    p_down = [
        "M 430 115 C 430 185, 160 195, 160 265",
        "M 480 115 L 480 265",
        "M 530 115 C 530 185, 800 195, 800 265"
    ]
    p_up = [
        "M 160 265 C 160 195, 430 185, 430 115",
        "M 480 265 L 480 115",
        "M 800 265 C 800 195, 530 195, 530 115"
    ]

    # Generate lines for each client depending on Online/Offline status and Phase
    line_elements = ""
    for idx in range(3):
        is_on = client_states[idx].get("online", True)
        if not is_on:
            line_elements += f"""
            <path d="{p_down[idx]}" class="line-offline" />
            <text x="{[280, 480, 680][idx]}" y="185" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#94a3b8" text-anchor="middle">❌ Terputus</text>
            """
        else:
            if phase == "BROADCAST":
                line_elements += f"""
                <path d="{p_down[idx]}" class="line-broadcast" marker-end="url(#arrow-blue)" />
                <circle r="6.5" fill="#3b82f6" filter="url(#glow-blue)">
                    <animateMotion dur="0.9s" repeatCount="indefinite" path="{p_down[idx]}" />
                </circle>
                <rect x="{[240, 440, 640][idx]}" y="172" width="80" height="19" rx="9" fill="#dbeafe" stroke="#93c5fd" />
                <text x="{[280, 480, 680][idx]}" y="185" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#1d4ed8" text-anchor="middle">Bobot W_t ↓</text>
                """
            elif phase == "UPLOAD_WEIGHTS":
                line_elements += f"""
                <path d="{p_up[idx]}" class="line-upload" marker-end="url(#arrow-orange)" />
                <circle r="6.5" fill="#f97316" filter="url(#glow-orange)">
                    <animateMotion dur="0.9s" repeatCount="indefinite" path="{p_up[idx]}" />
                </circle>
                <rect x="{[235, 435, 635][idx]}" y="172" width="90" height="19" rx="9" fill="#ffedd5" stroke="#fdba74" />
                <text x="{[280, 480, 680][idx]}" y="185" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#c2410c" text-anchor="middle">↑ Bobot ΔW_{idx+1}</text>
                """
            elif phase == "LOCAL_TRAIN":
                line_elements += f"""<path d="{p_down[idx]}" class="line-neutral" />"""
            elif phase == "FEDAVG_AGG":
                line_elements += f"""<path d="{p_down[idx]}" class="line-fedavg" />"""
            elif phase == "COMPLETED":
                # Alive continuous synchronization lines with dual particles (Emerald & Blue)
                line_elements += f"""
                <path d="{p_down[idx]}" class="line-sync" />
                <circle r="6" fill="#10b981" filter="url(#glow-green)">
                    <animateMotion dur="1.7s" repeatCount="indefinite" path="{p_down[idx]}" />
                </circle>
                <circle r="4.5" fill="#3b82f6" filter="url(#glow-blue)">
                    <animateMotion dur="1.7s" repeatCount="indefinite" path="{p_up[idx]}" />
                </circle>
                <rect x="{[235, 435, 635][idx]}" y="172" width="90" height="19" rx="9" fill="#d1fae5" stroke="#6ee7b7" />
                <text x="{[280, 480, 680][idx]}" y="185" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#065f46" text-anchor="middle">✨ Model Sinkron</text>
                """
            else:
                # READY / STANDBY: Subtle lively pulse line (not dead gray)
                line_elements += f"""
                <path d="{p_down[idx]}" class="line-ready" marker-end="url(#arrow-neutral)" />
                <circle r="4" fill="#93c5fd" opacity="0.8">
                    <animateMotion dur="2.4s" repeatCount="indefinite" path="{p_down[idx]}" />
                </circle>
                """

    # Generate 3 client node cards
    client_nodes_svg = ""
    client_x = [35, 355, 675]
    center_x = [160, 480, 800]

    for idx in range(3):
        c_state = client_states[idx]
        is_on = c_state.get("online", True)
        
        if not is_on:
            stroke = "#94a3b8"
            fill = "#f8fafc"
            opacity = "0.55"
            glow = ""
            badge_bg_c = "#f1f5f9"
            badge_color_c = "#64748b"
            badge_text_c = "🔴 OFFLINE / DISCONNECTED"
            metrics_text = "<tspan fill='#94a3b8'>Tidak Berpartisipasi</tspan>"
        else:
            opacity = "1.0"
            if phase == "LOCAL_TRAIN":
                stroke = "#10b981"
                fill = "#ecfdf5"
                glow = "filter: url(#glow-green); animation: pulseClient 0.9s infinite alternate;"
                badge_bg_c = "#d1fae5"
                badge_color_c = "#047857"
                badge_text_c = "🏋️ Training SGD Local..."
            elif phase == "UPLOAD_WEIGHTS":
                stroke = "#f97316"
                fill = "#fff7ed"
                glow = "filter: url(#glow-orange);"
                badge_bg_c = "#ffedd5"
                badge_color_c = "#c2410c"
                badge_text_c = "⬆️ Mengunggah Bobot (ΔW)"
            elif phase == "BROADCAST":
                stroke = "#3b82f6"
                fill = "#eff6ff"
                glow = ""
                badge_bg_c = "#dbeafe"
                badge_color_c = "#1d4ed8"
                badge_text_c = "📥 Menerima Bobot Global"
            elif phase == "COMPLETED":
                stroke = "#10b981"
                fill = "#f0fdf4"
                glow = "filter: url(#glow-green); animation: pulseCompleted 1.6s infinite alternate;"
                badge_bg_c = "#d1fae5"
                badge_color_c = "#065f46"
                badge_text_c = "✨ Sinkron dengan Model Global"
            else:
                stroke = "#64748b"
                fill = "#ffffff"
                glow = ""
                badge_bg_c = "#f1f5f9"
                badge_color_c = "#475569"
                badge_text_c = "🟢 Standby (Online)"

            c_acc = c_state.get("acc", 0.0)
            c_loss = c_state.get("loss", 0.0)
            metrics_text = f"Acc: <tspan fill='#047857' font-weight='800'>{c_acc:.1%}</tspan> | Loss: <tspan fill='#b91c1c' font-weight='800'>{c_loss:.4f}</tspan>"

        client_nodes_svg += f"""
        <g class="interactive-node" onclick="showNodeDetails({idx})" style="{glow} opacity: {opacity}; cursor: pointer;">
            <rect x="{client_x[idx]}" y="265" width="250" height="98" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.2" />
            <text x="{center_x[idx]}" y="291" font-family="system-ui, sans-serif" font-size="14.5" font-weight="800" fill="#0f172a" text-anchor="middle">📱 Client {idx+1} (SuperNode #{idx+1})</text>
            <text x="{center_x[idx]}" y="306" font-family="system-ui, sans-serif" font-size="10.5" fill="#64748b" text-anchor="middle">Partisi #{idx} (250 sampel)</text>
            
            <rect x="{client_x[idx]+25}" y="313" width="200" height="19" rx="9" fill="{badge_bg_c}" />
            <text x="{center_x[idx]}" y="326" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="{badge_color_c}" text-anchor="middle">{badge_text_c}</text>
            
            <text x="{center_x[idx]}" y="349" font-family="system-ui, sans-serif" font-size="11.5" font-weight="600" fill="#475569" text-anchor="middle">
                {metrics_text}
            </text>
        </g>
        """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: transparent;
        overflow: hidden;
        user-select: none;
    }}
    .phase-pill {{
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.2px;
        background: {badge_bg};
        color: {badge_color};
        border: 1.5px solid {badge_border};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .canvas-container {{
        position: relative;
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 8px;
        margin-top: 8px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.03);
    }}
    .interactive-node:hover rect {{
        stroke-width: 3.5px;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.14));
        transition: all 0.2s ease;
    }}
    @keyframes flowDown {{
        from {{ stroke-dashoffset: 32; }}
        to {{ stroke-dashoffset: 0; }}
    }}
    @keyframes flowUp {{
        from {{ stroke-dashoffset: 0; }}
        to {{ stroke-dashoffset: 32; }}
    }}
    @keyframes syncFlowDown {{
        from {{ stroke-dashoffset: 24; }}
        to {{ stroke-dashoffset: 0; }}
    }}
    @keyframes pulseClient {{
        0% {{ filter: drop-shadow(0 0 3px rgba(16, 185, 129, 0.4)); }}
        100% {{ filter: drop-shadow(0 0 14px rgba(16, 185, 129, 0.9)); }}
    }}
    @keyframes pulseServer {{
        0% {{ filter: drop-shadow(0 0 4px rgba(168, 85, 247, 0.4)); }}
        100% {{ filter: drop-shadow(0 0 18px rgba(245, 158, 11, 0.95)); }}
    }}
    @keyframes pulseCompleted {{
        0% {{ filter: drop-shadow(0 0 4px rgba(16, 185, 129, 0.4)); }}
        100% {{ filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.85)); }}
    }}
    .line-broadcast {{
        stroke: #3b82f6;
        stroke-width: 4;
        stroke-dasharray: 10, 6;
        animation: flowDown 0.75s linear infinite;
        fill: none;
    }}
    .line-upload {{
        stroke: #f97316;
        stroke-width: 4;
        stroke-dasharray: 10, 6;
        animation: flowUp 0.75s linear infinite;
        fill: none;
    }}
    .line-sync {{
        stroke: #10b981;
        stroke-width: 3.5;
        stroke-dasharray: 8, 4;
        animation: syncFlowDown 1.1s linear infinite;
        fill: none;
        filter: drop-shadow(0 0 4px rgba(16, 185, 129, 0.45));
    }}
    .line-ready {{
        stroke: #93c5fd;
        stroke-width: 2.2;
        stroke-dasharray: 6, 4;
        animation: flowDown 2.2s linear infinite;
        fill: none;
    }}
    .line-neutral {{
        stroke: #cbd5e1;
        stroke-width: 2.2;
        stroke-dasharray: 6, 6;
        fill: none;
    }}
    .line-offline {{
        stroke: #e2e8f0;
        stroke-width: 2;
        stroke-dasharray: 4, 4;
        fill: none;
    }}
    .line-fedavg {{
        stroke: #c084fc;
        stroke-width: 3;
        stroke-dasharray: 8, 4;
        fill: none;
    }}
    /* Modal Detail Card */
    .modal-overlay {{
        display: none;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 370px;
        background: white;
        border-radius: 14px;
        border: 2px solid #3b82f6;
        box-shadow: 0 16px 36px rgba(0,0,0,0.25);
        padding: 18px 22px;
        z-index: 100;
        animation: fadeIn 0.2s ease-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translate(-50%, -46%); }}
        to {{ opacity: 1; transform: translate(-50%, -50%); }}
    }}
    .close-btn-bottom {{
        background: #0f172a;
        color: white;
        border: none;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        cursor: pointer;
        transition: background 0.15s ease;
    }}
    .close-btn-bottom:hover {{ background: #1e293b; }}
</style>
</head>
<body>
    <div style="text-align: center;">
        <span class="phase-pill">{phase_text}</span>
    </div>

    <div class="canvas-container">
        <!-- Interactive Hint -->
        <div style="position: absolute; top: 12px; right: 16px; font-size: 0.76rem; color: #64748b; background: rgba(255,255,255,0.9); padding: 4px 10px; border-radius: 12px; border: 1px solid #e2e8f0; z-index: 10;">
            💡 <b>Tips:</b> Klik kotak Server atau Client untuk melihat detail teknis bobot & akurasi
        </div>

        <svg viewBox="0 0 960 380" width="100%" height="360" style="display: block;">
            <defs>
                <marker id="arrow-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#3b82f6" />
                </marker>
                <marker id="arrow-orange" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#f97316" />
                </marker>
                <marker id="arrow-neutral" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8" />
                </marker>

                <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
                <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
                <filter id="glow-green" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
                <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="6" result="blur" />
                    <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
            </defs>

            <!-- Connector Lines -->
            {line_elements}

            <!-- 1. Central Server Node (Top Center) -->
            <g class="interactive-node" onclick="showNodeDetails('server')" style="{srv_glow} cursor: pointer;">
                <rect x="330" y="16" width="300" height="98" rx="14" fill="{srv_fill}" stroke="{srv_stroke}" stroke-width="2.5" />
                <text x="480" y="42" font-family="system-ui, sans-serif" font-size="16" font-weight="800" fill="#0f172a" text-anchor="middle">🏢 Central Server (SuperLink)</text>
                
                <rect x="365" y="52" width="230" height="22" rx="11" fill="{srv_badge_bg}" />
                <text x="480" y="67" font-family="system-ui, sans-serif" font-size="10.8" font-weight="700" fill="{srv_badge_color}" text-anchor="middle">{srv_badge_text}</text>
                
                <text x="480" y="96" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">
                    Global Acc: <tspan fill="#047857" font-weight="800">{srv_acc:.1%}</tspan> | Global Loss: <tspan fill="#b91c1c" font-weight="800">{srv_loss:.4f}</tspan>
                </text>
            </g>

            <!-- 2, 3, 4: Client Nodes (Bottom Left, Center, Right) -->
            {client_nodes_svg}
        </svg>

        <!-- Interactive Popup Modal (Works in all phases including COMPLETED) -->
        <div id="detail-modal" class="modal-overlay">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 10px;">
                <h4 id="modal-title" style="margin:0; font-size:1.05rem; color:#0f172a;">Detail Node</h4>
                <button onclick="closeModal()" style="border:none; background:#f1f5f9; border-radius:50%; width:28px; height:28px; cursor:pointer; font-weight:bold; color:#64748b;">✕</button>
            </div>
            <div id="modal-content" style="font-size:0.86rem; color:#334155; line-height:1.55;"></div>
            <div style="text-align:right; margin-top:12px; border-top:1px solid #f1f5f9; padding-top:8px;">
                <button onclick="closeModal()" class="close-btn-bottom">Tutup</button>
            </div>
        </div>
    </div>

    <script>
        var nodeData = {{
            "server": {{
                "title": "🏢 Central Server (Flower SuperLink)",
                "status": "{server_state.get('status', 'Standby')}",
                "acc": "{srv_acc:.2%}",
                "loss": "{srv_loss:.4f}",
                "phase": "{phase}",
                "round": "{current_round}",
                "strategy": "Federated Averaging (FedAvg)",
                "model": "MLP (784 in -> 64 hidden -> 10 classes)",
                "weights_shape": "Layer 1: (64, 784), Layer 2: (10, 64)",
                "desc": "Mengorkestrasi siaran bobot model global dan melakukan rata-rata terbobot terhadap parameter gradien yang dikirim klien aktif."
            }},
            0: {{
                "title": "📱 Client 1 (SuperNode #1)",
                "online": "{client_states[0].get('online', True)}",
                "acc": "{client_states[0].get('acc', 0.0):.2%}",
                "loss": "{client_states[0].get('loss', 0.0):.4f}",
                "partition": "Partisi Data #0 (250 data latih, 125 data uji)",
                "optimizer": "SGD (Learning Rate: 0.05, Momentum: 0.9)",
                "payload": "Ukuran Bobot: ~203.5 KB per ronde"
            }},
            1: {{
                "title": "📱 Client 2 (SuperNode #2)",
                "online": "{client_states[1].get('online', True)}",
                "acc": "{client_states[1].get('acc', 0.0):.2%}",
                "loss": "{client_states[1].get('loss', 0.0):.4f}",
                "partition": "Partisi Data #1 (250 data latih, 125 data uji)",
                "optimizer": "SGD (Learning Rate: 0.05, Momentum: 0.9)",
                "payload": "Ukuran Bobot: ~203.5 KB per ronde"
            }},
            2: {{
                "title": "📱 Client 3 (SuperNode #3)",
                "online": "{client_states[2].get('online', True)}",
                "acc": "{client_states[2].get('acc', 0.0):.2%}",
                "loss": "{client_states[2].get('loss', 0.0):.4f}",
                "partition": "Partisi Data #2 (250 data latih, 125 data uji)",
                "optimizer": "SGD (Learning Rate: 0.05, Momentum: 0.9)",
                "payload": "Ukuran Bobot: ~203.5 KB per ronde"
            }}
        }};

        function showNodeDetails(nodeId) {{
            var info = nodeData[nodeId];
            if (!info) return;
            var modal = document.getElementById("detail-modal");
            var title = document.getElementById("modal-title");
            var content = document.getElementById("modal-content");
            
            title.innerHTML = info.title;
            if (nodeId === "server") {{
                var stateLabel = (info.phase === "COMPLETED") 
                    ? "<span style='color:#065f46; font-weight:bold;'>✨ Model Global Terdistribusi & Sinkron Sempurna</span>" 
                    : "<code>" + info.status + "</code>";
                content.innerHTML = `
                    <div style="margin-bottom:6px;"><b>Status Operasi:</b> ` + stateLabel + `</div>
                    <div style="margin-bottom:6px;"><b>Strategi Federasi:</b> ` + info.strategy + `</div>
                    <div style="margin-bottom:6px;"><b>Arsitektur Model:</b> ` + info.model + `</div>
                    <div style="margin-bottom:6px;"><b>Dimensi Bobot:</b> ` + info.weights_shape + `</div>
                    <div style="margin-bottom:6px;"><b>Akurasi Global:</b> <span style="color:#047857; font-weight:bold;">` + info.acc + `</span></div>
                    <div style="margin-bottom:8px;"><b>Loss Global:</b> <span style="color:#b91c1c; font-weight:bold;">` + info.loss + `</span></div>
                    <div style="font-size:0.8rem; color:#475569; background:#f8fafc; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">` + info.desc + `</div>
                `;
            }} else {{
                var statusBadge = info.online === "True" 
                    ? "<span style='color:#047857; font-weight:bold;'>🟢 Terhubung (Online)</span>" 
                    : "<span style='color:#b91c1c; font-weight:bold;'>🔴 Terputus (Offline)</span>";
                var syncLabel = (nodeData["server"].phase === "COMPLETED" && info.online === "True")
                    ? "<div style='color:#065f46; font-weight:bold; margin-bottom:6px;'>✨ Status: Model Sinkron dengan SuperLink</div>"
                    : "";
                content.innerHTML = `
                    <div style="margin-bottom:6px;"><b>Status Jaringan:</b> ` + statusBadge + `</div>
                    ` + syncLabel + `
                    <div style="margin-bottom:6px;"><b>Alokasi Data:</b> ` + info.partition + `</div>
                    <div style="margin-bottom:6px;"><b>Algoritma Lokal:</b> ` + info.optimizer + `</div>
                    <div style="margin-bottom:6px;"><b>Akurasi Lokal:</b> <span style="color:#047857; font-weight:bold;">` + info.acc + `</span></div>
                    <div style="margin-bottom:6px;"><b>Loss Lokal:</b> <span style="color:#b91c1c; font-weight:bold;">` + info.loss + `</span></div>
                    <div style="font-size:0.8rem; color:#475569; background:#f8fafc; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">` + info.payload + `</div>
                `;
            }}
            modal.style.display = "block";
        }}

        function closeModal() {{
            document.getElementById("detail-modal").style.display = "none";
        }}
    </script>
</body>
</html>
"""
    return full_html

# -----------------------------------------------------------------------------
# Top Placeholder for Zero-Scroll Real-Time Visual Diagram
# -----------------------------------------------------------------------------
topology_placeholder = st.empty()
status_placeholder = st.empty()
progress_placeholder = st.empty()

def update_topology(phase: str, r: int, total_r: int):
    """Render the topology inside the top placeholder without scrolling."""
    html_content = build_interactive_topology_html(
        phase=phase,
        server_state=st.session_state.server_state,
        client_states=st.session_state.client_states,
        current_round=r,
        total_rounds=total_r,
    )
    with topology_placeholder.container():
        components.html(html_content, height=430)

# Initial render
update_topology(
    phase=st.session_state.current_phase,
    r=st.session_state.current_round,
    total_r=num_rounds
)

# -----------------------------------------------------------------------------
# Metric Summary Cards
# -----------------------------------------------------------------------------
metric_placeholder = st.empty()

def update_metrics():
    curr_acc = st.session_state.server_state["acc"]
    curr_loss = st.session_state.server_state["loss"]
    has_prev = len(st.session_state.history) >= 2
    prev_acc = st.session_state.history[-2]["Global Acc"] if has_prev else curr_acc
    prev_loss = st.session_state.history[-2]["Global Loss"] if has_prev else curr_loss

    delta_acc = f"{(curr_acc - prev_acc):+.1%}" if has_prev else None
    delta_loss = f"{(curr_loss - prev_loss):+.4f}" if has_prev else None

    with metric_placeholder.container():
        m_cols = st.columns(4)
        m_cols[0].metric("Ronde Saat Ini", f"{st.session_state.current_round} / {num_rounds}")
        m_cols[1].metric("Akurasi Model Global", f"{curr_acc:.1%}", delta=delta_acc)
        m_cols[2].metric("Loss Model Global", f"{curr_loss:.4f}", delta=delta_loss, delta_color="inverse")
        m_cols[3].metric("SuperNodes Aktif", f"{num_online} / 3", delta=f"{num_online}/3 Online")

update_metrics()

# -----------------------------------------------------------------------------
# Dashboard Sections (Tabs)
# -----------------------------------------------------------------------------
tab_sim, tab_charts, tab_logs, tab_arch = st.tabs([
    "🌸 Live Console & Telemetry",
    "📈 Grafik Metrik Loss & Akurasi",
    "📋 Riwayat Ronde Federasi",
    "ℹ️ Arsitektur Flower SuperLink"
])

with tab_sim:
    st.markdown("#### 💻 Flower SuperLink Event Stream (Live Console)")
    log_box_placeholder = st.empty()

    def render_logs():
        log_html = "<div class='terminal-box'>"
        for line in st.session_state.logs[-8:]:
            log_html += f"<div class='terminal-line'>{line}</div>"
        log_html += "</div>"
        log_box_placeholder.markdown(log_html, unsafe_allow_html=True)

    render_logs()

with tab_charts:
    charts_placeholder = st.empty()

    def update_charts():
        with charts_placeholder.container():
            c_col1, c_col2 = st.columns(2)
            if st.session_state.history:
                df_hist = pd.DataFrame(st.session_state.history)
                with c_col1:
                    st.markdown("#### 📈 Progres Akurasi Global & Klien")
                    acc_cols = [c for c in ["Global Acc", "Client 1 Acc", "Client 2 Acc", "Client 3 Acc"] if c in df_hist.columns]
                    st.line_chart(df_hist.set_index("Round")[acc_cols])
                with c_col2:
                    st.markdown("#### 📉 Konvergensi Loss Global & Klien")
                    loss_cols = [c for c in ["Global Loss", "Client 1 Loss", "Client 2 Loss", "Client 3 Loss"] if c in df_hist.columns]
                    st.line_chart(df_hist.set_index("Round")[loss_cols])
            else:
                with c_col1:
                    st.info("💡 Grafik akurasi akan otomatis terisi secara dinamis saat simulasi berjalan.")
                with c_col2:
                    st.info("💡 Grafik konvergensi loss akan menampilkan proses penurunan error.")

    update_charts()

with tab_logs:
    history_table_placeholder = st.empty()

    def update_history_table():
        if st.session_state.history:
            df_display = pd.DataFrame(st.session_state.history)
            formatted_df = df_display.copy()
            for col in formatted_df.columns:
                if "Acc" in col:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}")
                elif "Loss" in col:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
            history_table_placeholder.dataframe(formatted_df, use_container_width=True)
        else:
            history_table_placeholder.info("Belum ada ronde federasi yang selesai dijalankan.")

    update_history_table()

with tab_arch:
    st.markdown("""
    ### 🌸 Arsitektur Flower Federated Learning
    
    1. **Flower SuperLink (Central Server)**:
       - Bertindak sebagai simpul koordinasi pusat (*orchestrator*).
       - Menyiarkan parameter bobot model global $W_t$ kepada klien aktif.
       - Menerima pembaruan gradien/bobot dari klien ($\Delta W_k$).
       - Mengagregasi bobot secara adaptif dari klien online menggunakan **FedAvg**:
         $$W_{t+1} = \\sum_{k \\in \\text{Active}} \\frac{n_k}{N_{\\text{active}}} W_{t+1}^k$$

    2. **Flower SuperNodes (Clients 1, 2, 3)**:
       - Melakukan pelatihan lokal (*Local SGD*) pada dataset partisi privat tanpa membagikan data mentah.
       - Saat ronde selesai, seluruh node yang aktif tetap berada dalam kondisi **sinkron** dengan model global terbaru.
    """)

# -----------------------------------------------------------------------------
# Core Federated Learning Execution Function (1 Round Engine)
# -----------------------------------------------------------------------------
def run_single_round(round_idx: int, total_rounds: int, active_flags: list):
    """Executes exactly 1 federated learning round across active clients."""
    
    active_indices = [i for i, on in enumerate(active_flags) if on]
    if not active_indices:
        st.session_state.logs.append(
            f"[{time.strftime('%H:%M:%S')}] <span class='term-warn'>[Peringatan]</span> Semua klien offline! Silakan nyalakan minimal 1 klien untuk melanjutkan ronde."
        )
        render_logs()
        status_placeholder.warning("⚠️ Semua klien dalam kondisi offline. Harap aktifkan minimal 1 klien melalui saklar.")
        return False

    partitions_train, partitions_test = st.session_state.partitions_cache
    global_model = st.session_state.global_model
    current_global_weights = st.session_state.global_weights

    # Combined test data
    all_test_x = np.concatenate([pt[0] for pt in partitions_test], axis=0)
    all_test_y = np.concatenate([pt[1] for pt in partitions_test], axis=0)
    
    if not HAS_MLX:
        test_ds = TensorDataset(torch.from_numpy(all_test_x), torch.from_numpy(all_test_y))
        test_loader_global = DataLoader(test_ds, batch_size=32, shuffle=False)
    else:
        test_loader_global = (all_test_x, all_test_y)

    # Initial evaluation if round 0
    if len(st.session_state.history) == 0:
        init_loss, init_acc = test(global_model, test_loader_global)
        st.session_state.server_state = {"status": "Initialized (Round 0)", "loss": init_loss, "acc": init_acc}
        st.session_state.history.append({
            "Round": 0,
            "Global Acc": init_acc,
            "Global Loss": init_loss,
            "Client 1 Acc": init_acc,
            "Client 1 Loss": init_loss,
            "Client 2 Acc": init_acc,
            "Client 2 Loss": init_loss,
            "Client 3 Acc": init_acc,
            "Client 3 Loss": init_loss,
        })
        update_metrics()
        update_charts()
        update_history_table()

    st.session_state.current_round = round_idx
    progress_placeholder.progress(min(1.0, round_idx / total_rounds))

    # -------------------------------------------------------------
    # Phase 1: Broadcast
    # -------------------------------------------------------------
    st.session_state.current_phase = "BROADCAST"
    st.session_state.server_state["status"] = f"Broadcasting weights W_{round_idx} to Active Clients"
    for idx in range(3):
        if active_flags[idx]:
            st.session_state.client_states[idx]["status"] = "Menerima bobot global..."
        else:
            st.session_state.client_states[idx]["status"] = "Offline (Terputus)"

    update_topology("BROADCAST", round_idx, total_rounds)
    active_str = ", ".join([f"SuperNode #{i+1}" for i in active_indices])
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-srv'>[SuperLink]</span> 📡 <b>Ronde {round_idx}/{total_rounds}</b>: Menyiarkan bobot global ke {active_str}."
    )
    render_logs()
    status_placeholder.markdown(f"**[Ronde {round_idx}/{total_rounds}] Fase 1:** Server menyiarkan bobot global ke klien online ({active_str})...")
    time.sleep(anim_delay)

    # -------------------------------------------------------------
    # Phase 2: Local Training
    # -------------------------------------------------------------
    st.session_state.current_phase = "LOCAL_TRAIN"
    st.session_state.server_state["status"] = "Clients training locally..."
    for idx in range(3):
        if active_flags[idx]:
            st.session_state.client_states[idx]["status"] = f"Training locally (Epochs: {local_epochs})..."
        else:
            st.session_state.client_states[idx]["status"] = "Offline (Tidak Melatih)"

    update_topology("LOCAL_TRAIN", round_idx, total_rounds)
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-cli'>[SuperNodes]</span> 🏋️ {len(active_indices)} Klien aktif melatih model secara lokal (Epochs: {local_epochs}, LR: {learning_rate})."
    )
    render_logs()
    status_placeholder.markdown(f"**[Ronde {round_idx}/{total_rounds}] Fase 2:** {len(active_indices)} SuperNode Klien melatih data privat secara independen...")

    client_weights = {}
    client_metrics = {}

    for idx in active_indices:
        client_model = load_model()
        set_weights(client_model, current_global_weights)

        c_train_x, c_train_y = partitions_train[idx]
        c_test_x, c_test_y = partitions_test[idx]

        if not HAS_MLX:
            train_ds = TensorDataset(torch.from_numpy(c_train_x), torch.from_numpy(c_train_y))
            train_loader_c = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_ds = TensorDataset(torch.from_numpy(c_test_x), torch.from_numpy(c_test_y))
            test_loader_c = DataLoader(test_ds, batch_size=32, shuffle=False)
            c_train_data = train_loader_c
            c_test_data = test_loader_c
        else:
            c_train_data = (c_train_x, c_train_y)
            c_test_data = (c_test_x, c_test_y)

        # Local training
        train(client_model, c_train_data, epochs=local_epochs, lr=learning_rate)
        c_loss, c_acc = test(client_model, c_test_data)
        
        st.session_state.client_states[idx]["loss"] = c_loss
        st.session_state.client_states[idx]["acc"] = c_acc
        st.session_state.client_states[idx]["status"] = "Training selesai"

        client_weights[idx] = get_weights(client_model)
        client_metrics[idx] = {"loss": c_loss, "accuracy": c_acc}

    time.sleep(anim_delay)

    # -------------------------------------------------------------
    # Phase 3: Upload Weights
    # -------------------------------------------------------------
    st.session_state.current_phase = "UPLOAD_WEIGHTS"
    st.session_state.server_state["status"] = "Receiving updated weights from SuperNodes..."
    for idx in range(3):
        if active_flags[idx]:
            st.session_state.client_states[idx]["status"] = "Mengunggah bobot -> Server"
        else:
            st.session_state.client_states[idx]["status"] = "Offline"

    update_topology("UPLOAD_WEIGHTS", round_idx, total_rounds)
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-cli'>[SuperNodes]</span> ⬆️ Mengunggah bobot gradien dari {len(active_indices)} klien aktif -> SuperLink."
    )
    render_logs()
    status_placeholder.markdown(f"**[Ronde {round_idx}/{total_rounds}] Fase 3:** Mengunggah bobot hasil training lokal dari klien aktif ke Server...")
    time.sleep(anim_delay)

    # -------------------------------------------------------------
    # Phase 4: FedAvg Aggregation
    # -------------------------------------------------------------
    st.session_state.current_phase = "FEDAVG_AGG"
    st.session_state.server_state["status"] = "Aggregating via Federated Averaging (FedAvg)..."
    for idx in range(3):
        if active_flags[idx]:
            st.session_state.client_states[idx]["status"] = "Standby (Menunggu Ronde Berikutnya)"

    update_topology("FEDAVG_AGG", round_idx, total_rounds)
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-fed'>[FedAvg]</span> 🧮 Server mengagregasi parameter dari {len(active_indices)} klien online secara adaptif..."
    )
    render_logs()
    status_placeholder.markdown(f"**[Ronde {round_idx}/{total_rounds}] Fase 4:** Server mengagregasi parameter menggunakan algoritma Federated Averaging (FedAvg)...")

    # Federated Averaging across active clients
    aggregated_weights = []
    sample_key = list(client_weights.keys())[0]
    num_layers = len(client_weights[sample_key])
    n_active = len(active_indices)

    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(client_weights[sample_key][layer_idx], dtype=np.float32)
        for idx in active_indices:
            layer_sum += client_weights[idx][layer_idx]
        aggregated_weights.append(layer_sum / float(n_active))

    # Update global model
    set_weights(global_model, aggregated_weights)
    st.session_state.global_weights = aggregated_weights

    # Evaluate global model
    global_loss, global_acc = test(global_model, test_loader_global)
    st.session_state.server_state["loss"] = global_loss
    st.session_state.server_state["acc"] = global_acc
    st.session_state.server_state["status"] = f"Ronde {round_idx} Selesai"

    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-fed'>[Evaluasi]</span> Ronde {round_idx} tuntas! Akurasi Global: <b>{global_acc:.1%}</b>, Loss: <b>{global_loss:.4f}</b>"
    )
    render_logs()

    # Log history
    st.session_state.history.append({
        "Round": round_idx,
        "Global Acc": global_acc,
        "Global Loss": global_loss,
        "Client 1 Acc": client_metrics.get(0, {"accuracy": 0.0})["accuracy"] if active_flags[0] else None,
        "Client 1 Loss": client_metrics.get(0, {"loss": 0.0})["loss"] if active_flags[0] else None,
        "Client 2 Acc": client_metrics.get(1, {"accuracy": 0.0})["accuracy"] if active_flags[1] else None,
        "Client 2 Loss": client_metrics.get(1, {"loss": 0.0})["loss"] if active_flags[1] else None,
        "Client 3 Acc": client_metrics.get(2, {"accuracy": 0.0})["accuracy"] if active_flags[2] else None,
        "Client 3 Loss": client_metrics.get(2, {"loss": 0.0})["loss"] if active_flags[2] else None,
    })

    update_metrics()
    update_charts()
    update_history_table()
    time.sleep(anim_delay)
    return True

# -----------------------------------------------------------------------------
# Trigger: Putar Ulang Animasi Visual (Demo Loop)
# -----------------------------------------------------------------------------
if btn_demo:
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-sync'>[Demo Loop]</span> 🎬 Memutar ulang visualisasi alur 4 fase interaksi Server & Klien..."
    )
    render_logs()
    
    demo_phases = [
        ("BROADCAST", "📡 [Demo] FASE 1: Broadcast — Server mengirim Bobot Global W_t ke Klien (Garis Biru Meluncur Turun)"),
        ("LOCAL_TRAIN", "🏋️ [Demo] FASE 2: Local Training — SuperNode Klien melatih data secara lokal (Glow Pulsasi Hijau)"),
        ("UPLOAD_WEIGHTS", "⬆️ [Demo] FASE 3: Upload Weights — Klien mengunggah bobot gradien ke Server (Garis Oranye Meluncur Naik)"),
        ("FEDAVG_AGG", "🧮 [Demo] FASE 4: FedAvg Aggregation — Server mengagregasi parameter model (Glow Ungu-Emas)"),
        ("COMPLETED", "✨ [Demo] Sinkronisasi Selesai — Seluruh node kini sinkron dengan aliran partikel hijau-biru berkelanjutan"),
    ]

    r_num = max(1, st.session_state.current_round)
    for d_phase, d_text in demo_phases:
        st.session_state.current_phase = d_phase
        update_topology(d_phase, r_num, num_rounds)
        status_placeholder.markdown(f"**{d_text}**")
        time.sleep(max(0.6, anim_delay))
        
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] <span class='term-sync'>[Demo Loop]</span> ✨ Putar ulang animasi selesai! Kanvas tetap dalam mode sinkronisasi berkelanjutan."
    )
    render_logs()
    st.session_state.current_phase = "COMPLETED"
    update_topology("COMPLETED", r_num, num_rounds)

# -----------------------------------------------------------------------------
# Trigger: Step-by-Step Execution (1 Round)
# -----------------------------------------------------------------------------
if btn_step:
    next_round = st.session_state.current_round + 1
    if next_round > num_rounds:
        status_placeholder.info(f"💡 Target {num_rounds} ronde sudah tercapai. Klik '🔄 Reset' atau naikkan target ronde di sidebar untuk menambah ronde.")
    else:
        success = run_single_round(next_round, num_rounds, active_clients)
        if success:
            if next_round == num_rounds:
                st.session_state.current_phase = "COMPLETED"
                update_topology("COMPLETED", num_rounds, num_rounds)
                status_placeholder.success(f"🎉 Target {num_rounds} ronde selesai! Model Global kini terdistribusi dan sinkron dengan akurasi: {st.session_state.server_state['acc']:.1%}")
            else:
                st.session_state.current_phase = "STANDBY"
                update_topology("READY", next_round, num_rounds)
                status_placeholder.info(f"⏸️ Ronde {next_round} selesai. Klik '⏭️ Jalankan 1 Ronde' lagi untuk ronde {next_round + 1}.")

# -----------------------------------------------------------------------------
# Trigger: Automatic Execution (All remaining rounds)
# -----------------------------------------------------------------------------
if btn_auto:
    start_r = st.session_state.current_round + 1
    if start_r > num_rounds:
        start_r = 1
        st.session_state.current_round = 0
        st.session_state.history = []

    for r in range(start_r, num_rounds + 1):
        success = run_single_round(r, num_rounds, active_clients)
        if not success:
            break

    if st.session_state.current_round == num_rounds:
        st.session_state.current_phase = "COMPLETED"
        update_topology("COMPLETED", num_rounds, num_rounds)
        final_acc = st.session_state.server_state["acc"]
        status_placeholder.success(f"🎉 Seluruh {num_rounds} ronde selesai secara otomatis! Model global sinkron sempurna dengan akurasi akhir: {final_acc:.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Flower 1.x Architecture • Quickstart MLX with PyTorch Fallback • Streamlit Community Cloud Ready
</div>
""", unsafe_allow_html=True)
