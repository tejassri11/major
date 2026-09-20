"""
tls_editor.py — Visual TLS phase editor for SUMO
Run: streamlit run tls_editor.py
"""
import streamlit as st
import plotly.graph_objects as go
import os

st.set_page_config(page_title="TLS Phase Editor", layout="wide", page_icon="🚦")

# ── Intersection definitions ──────────────────────────────────────────────────
TLS = {
    "TL_A": {
        "id": "6073919354",
        "n_links": 10,
        "arms": {
            "W  -E5":            [0, 1, 2],
            "N  -465932558":     [3, 4, 5],
            "E  470773638":      [6, 7],
            "S  465932558#0":    [8, 9],
        },
        "dirs": {0:"L", 1:"S", 2:"R", 3:"L", 4:"S", 5:"R", 6:"S", 7:"R", 8:"S", 9:"R"},
        "arm_dir": {"W  -E5":"W", "N  -465932558":"N", "E  470773638":"E", "S  465932558#0":"S"},
    },
    "TL_B": {
        "id": "6073919354_B",
        "n_links": 10,
        "arms": {
            "W  -E5_B":           [0, 1, 2],
            "N  -465932558_B":    [3, 4, 5],
            "E  470773638_B":     [6, 7],
            "S  465932558#0_B":   [8, 9],
        },
        "dirs": {0:"L", 1:"S", 2:"R", 3:"L", 4:"S", 5:"R", 6:"S", 7:"R", 8:"S", 9:"R"},
        "arm_dir": {"W  -E5_B":"W", "N  -465932558_B":"N", "E  470773638_B":"E", "S  465932558#0_B":"S"},
    },
    "TL_C": {
        "id": "6073919354_C",
        "n_links": 6,
        "arms": {
            "N  -465932558#2":   [0, 1],
            "E  470773638#1":    [2, 3],
            "W  E0":             [4, 5],
        },
        "dirs": {0:"S", 1:"R", 2:"L", 3:"R", 4:"L", 5:"S"},
        "arm_dir": {"N  -465932558#2":"N", "E  470773638#1":"E", "W  E0":"W"},
    },
}

STATES     = ["r", "G", "g", "y"]
S_COLOR    = {"r": "#ff3b30", "G": "#34c759", "g": "#5ac882", "y": "#ff9500"}
S_EMOJI    = {"r": "🔴", "G": "🟢", "g": "🟩", "y": "🟡"}
S_MEANING  = {"r": "red", "G": "protected green", "g": "yield green", "y": "yellow"}
DIR_SYM    = {"L": "↰", "S": "↑", "R": "↱"}

NETWORK_TLS = "network/tls.add.xml"

# ── Session state helpers ─────────────────────────────────────────────────────
def sk(tl, ph, lk):   return f"s_{tl}_{ph}_{lk}"
def dk(tl, ph):        return f"d_{tl}_{ph}"
def nk(tl):            return f"n_{tl}"

def ensure_defaults(tl_key, n_phases):
    n_links = TLS[tl_key]["n_links"]
    for p in range(n_phases):
        for l in range(n_links):
            k = sk(tl_key, p, l)
            if k not in st.session_state:
                st.session_state[k] = "r"
        if dk(tl_key, p) not in st.session_state:
            st.session_state[dk(tl_key, p)] = 36 if p % 2 == 0 else 4

def get_state_str(tl_key, phase):
    n = TLS[tl_key]["n_links"]
    return "".join(st.session_state.get(sk(tl_key, phase, l), "r") for l in range(n))

def load_from_xml(path):
    """Parse existing tls.add.xml and push values into session_state."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    id_to_key = {v["id"]: k for k, v in TLS.items()}
    for logic in root.findall("tlLogic"):
        tl_id  = logic.get("id")
        tl_key = id_to_key.get(tl_id)
        if not tl_key:
            continue
        phases = logic.findall("phase")
        st.session_state[nk(tl_key)] = len(phases)
        for pi, ph in enumerate(phases):
            state = ph.get("state", "")
            dur   = int(ph.get("duration", 36))
            st.session_state[dk(tl_key, pi)] = dur
            for li, ch in enumerate(state):
                if ch in STATES:
                    st.session_state[sk(tl_key, pi, li)] = ch

# ── Intersection diagram ──────────────────────────────────────────────────────
ARM_CFG = {
    "N": dict(sx=0,    sy=3,    ex=0,  ey=1,  perp_x=1,  perp_y=0),
    "S": dict(sx=0,    sy=-3,   ex=0,  ey=-1, perp_x=1,  perp_y=0),
    "E": dict(sx=3,    sy=0,    ex=1,  ey=0,  perp_x=0,  perp_y=1),
    "W": dict(sx=-3,   sy=0,    ex=-1, ey=0,  perp_x=0,  perp_y=1),
}

def make_diagram(tl_key):
    tl  = TLS[tl_key]
    fig = go.Figure()

    # Intersection box
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1,
                  fillcolor="#e5e5ea", line_color="#6e6e73", line_width=2)
    fig.add_annotation(x=0, y=0, text="<b>⬛</b>", showarrow=False,
                       font=dict(size=18, color="#6e6e73"))

    for arm_label, links in tl["arms"].items():
        arm_dir = tl["arm_dir"][arm_label]
        if arm_dir not in ARM_CFG:
            continue
        c   = ARM_CFG[arm_dir]
        n   = len(links)
        for i, link_idx in enumerate(links):
            offset = (i - (n - 1) / 2) * 0.45
            x0 = c["sx"] + c["perp_x"] * offset
            y0 = c["sy"] + c["perp_y"] * offset
            x1 = c["ex"] + c["perp_x"] * offset
            y1 = c["ey"] + c["perp_y"] * offset
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2

            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.5, arrowwidth=2,
                arrowcolor="#0071e3", showarrow=True, text="",
            )
            dsym = DIR_SYM.get(tl["dirs"][link_idx], "")
            fig.add_annotation(
                x=mx, y=my,
                text=f"<b>{link_idx}</b> {dsym}",
                showarrow=False,
                bgcolor="white", bordercolor="#0071e3",
                borderwidth=1, borderpad=3,
                font=dict(size=11, color="#0071e3"),
            )

        # Arm label
        lx = c["sx"] * 0.85 + c["perp_x"] * ((n - 1) / 2 + 0.8) * 0.45
        ly = c["sy"] * 0.85 + c["perp_y"] * ((n - 1) / 2 + 0.8) * 0.45
        fig.add_annotation(x=lx, y=ly, text=f"<i>{arm_dir}</i>",
                           showarrow=False, font=dict(size=10, color="#6e6e73"))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-3.8, 3.8], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-3.8, 3.8], showgrid=False, zeroline=False, showticklabels=False),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
    )
    return fig

# ── Colored state string badge ────────────────────────────────────────────────
def colored_state(s):
    badges = "".join(
        f'<span style="background:{S_COLOR[c]};color:white;padding:3px 7px;'
        f'border-radius:5px;font-family:monospace;font-size:14px;margin:2px;">'
        f'{c}</span>'
        for c in s
    )
    return f'<div style="margin:6px 0">{badges}</div>'

# ── XML builder ───────────────────────────────────────────────────────────────
def build_xml(tl_phases):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<additional>']
    for tl_key, phases in tl_phases.items():
        tl_id = TLS[tl_key]["id"]
        lines.append(f'')
        lines.append(f'    <tlLogic id="{tl_id}" type="static" programID="rl_program" offset="0">')
        for pi, (state, dur) in enumerate(phases):
            lines.append(f'        <phase duration="{dur}" state="{state}"/>')
        lines.append(f'    </tlLogic>')
    lines += ['', '</additional>']
    return "\n".join(lines)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚦 TLS Phase Editor")
st.caption("Design SUMO traffic light phase programs — generates `network/tls.add.xml` directly")

# Load existing file
c1, c2, _ = st.columns([1, 1, 4])
if c1.button("📂 Load network/tls.add.xml"):
    if os.path.exists(NETWORK_TLS):
        load_from_xml(NETWORK_TLS)
        st.success("Loaded current tls.add.xml")
    else:
        st.warning("File not found")

if c2.button("🗑 Reset all"):
    for k in list(st.session_state.keys()):
        if k.startswith(("s_", "d_", "n_")):
            del st.session_state[k]
    st.rerun()

st.divider()

tl_tabs = st.tabs(list(TLS.keys()))
all_phases = {}

for tab, tl_key in zip(tl_tabs, TLS.keys()):
    with tab:
        tl = TLS[tl_key]

        if nk(tl_key) not in st.session_state:
            st.session_state[nk(tl_key)] = 6 if tl_key == "TL_C" else 4

        left, right = st.columns([1, 2])

        with left:
            st.plotly_chart(make_diagram(tl_key), use_container_width=True)
            st.caption("Arrows = incoming link directions.  Number = link index.")

        with right:
            n_phases = st.number_input(
                "Number of phases", min_value=2, max_value=14,
                step=2, key=nk(tl_key),
            )
            ensure_defaults(tl_key, n_phases)

            ph_tabs = st.tabs([f"Phase {i}" for i in range(n_phases)])
            phases_out = []

            for pi, ptab in enumerate(ph_tabs):
                with ptab:
                    dur = st.number_input("Duration (s)", 1, 180, key=dk(tl_key, pi))

                    for arm_label, links in tl["arms"].items():
                        st.markdown(f"**{arm_label}**")
                        cols = st.columns(len(links))
                        for col, link_idx in zip(cols, links):
                            dsym = DIR_SYM.get(tl["dirs"][link_idx], "")
                            cur  = st.session_state.get(sk(tl_key, pi, link_idx), "r")
                            col.selectbox(
                                f"**{link_idx}** {dsym}",
                                options=STATES,
                                index=STATES.index(cur),
                                key=sk(tl_key, pi, link_idx),
                                format_func=lambda s: f"{S_EMOJI[s]} {s}  ({S_MEANING[s]})",
                            )

                    s = get_state_str(tl_key, pi)
                    st.markdown(colored_state(s), unsafe_allow_html=True)
                    st.code(f'<phase duration="{dur}" state="{s}"/>', language="xml")
                    phases_out.append((s, dur))

        all_phases[tl_key] = phases_out

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
xml = build_xml(all_phases)

ea, eb, _ = st.columns([1, 1, 4])
ea.download_button("⬇ Download tls.add.xml", xml, "tls.add.xml", "text/xml",
                   use_container_width=True)
if eb.button("💾 Save → network/tls.add.xml", use_container_width=True):
    os.makedirs("network", exist_ok=True)
    with open(NETWORK_TLS, "w", encoding="utf-8") as f:
        f.write(xml)
    st.success(f"Saved to {NETWORK_TLS}")

with st.expander("Preview XML"):
    st.code(xml, language="xml")
