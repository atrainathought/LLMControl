"""
Shared styles for LLMControl Streamlit app.
Dark mode compatible.
"""

CUSTOM_CSS = """
<style>
    /* ===== DARK MODE COMPATIBLE STYLES ===== */

    /* CSS Variables for theming */
    :root {
        --primary: #667eea;
        --primary-light: #8b9df0;
        --secondary: #764ba2;
        --accent: #a855f7;
    }

    /* Header styling - works in both modes */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #8b9df0 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    /* Sub-header adapts to theme */
    .sub-header {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Section dividers - theme aware */
    .section-divider {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0.2;
    }

    /* ===== TABS - Dark mode fix ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
        background-color: transparent;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* ===== BUTTONS - Dark mode fix ===== */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(90deg, #5a6fd6 0%, #6a4190 100%) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        border: 1px solid #667eea !important;
        color: #667eea !important;
        background: transparent !important;
    }

    /* ===== SIDEBAR - Dark mode fix ===== */
    [data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.05);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.9;
    }

    /* Sidebar navigation links */
    [data-testid="stSidebar"] a {
        color: inherit !important;
        opacity: 0.8;
    }

    [data-testid="stSidebar"] a:hover {
        opacity: 1;
    }

    /* ===== INPUTS - Dark mode fix ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        border-radius: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }

    /* ===== EXPANDERS - Dark mode fix ===== */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 0.5rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
    }

    /* ===== METRICS - Dark mode fix ===== */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
    }

    [data-testid="stMetricDeltaIcon-Up"] {
        color: #22c55e;
    }

    [data-testid="stMetricDeltaIcon-Down"] {
        color: #ef4444;
    }

    /* ===== DATAFRAMES - Dark mode fix ===== */
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* ===== ALERTS - Dark mode fix ===== */
    .stAlert {
        border-radius: 0.5rem;
    }

    /* Success alert */
    [data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 0.5rem;
    }

    /* ===== CODE BLOCKS ===== */
    .stCodeBlock {
        border-radius: 0.5rem;
    }

    /* ===== TOGGLE/CHECKBOX - Dark mode fix ===== */
    .stCheckbox label span,
    .stToggle label span {
        opacity: 0.9;
    }

    /* ===== SELECTBOX - Dark mode fix ===== */
    [data-baseweb="select"] {
        border-radius: 0.5rem;
    }

    [data-baseweb="popover"] {
        border-radius: 0.5rem;
    }

    /* ===== SLIDER - Dark mode fix ===== */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #667eea;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* ===== JSON OUTPUT ===== */
    .stJson {
        border-radius: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* ===== CONTAINER SPACING ===== */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.75rem;
        }
        .sub-header {
            font-size: 1rem;
        }
    }

    /* ===== PLOTLY CHARTS - Dark mode fix ===== */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    /* ===== RADIO BUTTONS - Dark mode fix ===== */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio label {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
        cursor: pointer;
        transition: all 0.2s;
    }

    .stRadio label:hover {
        border-color: #667eea;
    }

    .stRadio [data-checked="true"] label {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }

    /* ===== NUMBER INPUT BUTTONS ===== */
    .stNumberInput button {
        background: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
    }

    .stNumberInput button:hover {
        border-color: #667eea !important;
    }
</style>
"""


def apply_styles():
    """Apply custom styles to the page."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = ""):
    """Render a consistent page header."""
    import streamlit as st
    if icon:
        st.markdown(f'<p class="main-header">{icon} {title}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="main-header">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{subtitle}</p>', unsafe_allow_html=True)


def section_divider():
    """Render a styled section divider."""
    import streamlit as st
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def info_card(content: str):
    """Render a styled info card."""
    import streamlit as st
    st.markdown(f'<div class="info-card">{content}</div>', unsafe_allow_html=True)


def setup_sidebar():
    """Setup consistent sidebar with API key input."""
    import streamlit as st

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Settings")
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Optional - enables live LLM calls"
        )
        if api_key:
            st.session_state["api_key"] = api_key
            st.success("API key set!")

        st.markdown("---")
        st.caption("LLMControl v1.0")
