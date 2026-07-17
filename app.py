"""
TM2-RAG: Temporal Multimodal Medical RAG — Streamlit Chatbot

A ChatGPT/Gemini-style conversational interface for chest X-ray
report generation with temporal reasoning and Grad-CAM explainability.

Run:  streamlit run app.py
"""

import sys
import socket
import tempfile
import time
import re
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def get_lan_url(port: int = 8501) -> str:
    """Best-effort LAN URL for opening the app from another device."""
    host = "localhost"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
    except OSError:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            host = "localhost"
    return f"http://{host}:{port}"


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TM2-RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — ChatGPT / Gemini-style dark theme
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ───────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit chrome while keeping sidebar toggle visible ── */
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0;
        box-sizing: border-box;
        width: min(52rem, 100%);
        max-width: 52rem;
        margin-left: auto;
        margin-right: auto;
    }

    div[data-testid="stMainBlockContainer"] {
        box-sizing: border-box !important;
        width: min(56rem, 100%) !important;
        max-width: 56rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    section[data-testid="stMain"] {
        display: flex !important;
        justify-content: center !important;
    }

    section[data-testid="stMain"] > div {
        width: 100% !important;
    }

    section[data-testid="stMain"] .block-container,
    section.main .block-container {
        margin-left: auto;
        margin-right: auto;
    }

    /* ── Chat input bar ──────────────────────────────────── */
    .stChatInput {
        border-radius: 24px !important;
    }

    .stChatInput > div {
        border-radius: 24px !important;
    }

    /* ── Upload area ─────────────────────────────────────── */
    .upload-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        align-items: flex-start;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 340px;
    }

    /* Keep the sidebar toggle as a small left-edge rail, not a floating arrow. */
    div[data-testid="stCollapsedControl"],
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapsedControl"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"],
    button[aria-label="Expand sidebar"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Show sidebar"],
    button[aria-label="Hide sidebar"],
    button[title="Open sidebar"],
    button[title="Close sidebar"],
    button[title="Expand sidebar"],
    button[title="Collapse sidebar"],
    button[title="Show sidebar"],
    button[title="Hide sidebar"] {
        position: fixed !important;
        top: 0.75rem !important;
        left: 0 !important;
        right: auto !important;
        transform: none !important;
        z-index: 999999 !important;
        width: 2rem !important;
        height: 2rem !important;
        min-width: 2rem !important;
        padding: 0 !important;
        border-radius: 0 8px 8px 0 !important;
        background: rgba(35, 38, 47, 0.92) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-left: 0 !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.18) !important;
        overflow: hidden !important;
    }

    div[data-testid="stCollapsedControl"] svg,
    div[data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="stSidebarCollapsedControl"] svg,
    button[aria-label="Open sidebar"] svg,
    button[aria-label="Close sidebar"] svg,
    button[aria-label="Expand sidebar"] svg,
    button[aria-label="Collapse sidebar"] svg,
    button[aria-label="Show sidebar"] svg,
    button[aria-label="Hide sidebar"] svg,
    button[title="Open sidebar"] svg,
    button[title="Close sidebar"] svg,
    button[title="Expand sidebar"] svg,
    button[title="Collapse sidebar"] svg,
    button[title="Show sidebar"] svg,
    button[title="Hide sidebar"] svg {
        width: 0.9rem !important;
        height: 0.9rem !important;
        margin: auto !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        pointer-events: none !important;
    }

    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] div[data-testid="stCollapsedControl"] {
        pointer-events: auto !important;
    }

    header[data-testid="stHeader"] button:has(svg),
    header[data-testid="stHeader"] div:has(> button svg) {
        position: fixed !important;
        top: 0.75rem !important;
        left: 0 !important;
        right: auto !important;
        transform: none !important;
        z-index: 999999 !important;
        width: 2rem !important;
        height: 2rem !important;
        min-width: 2rem !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 0 8px 8px 0 !important;
        background: rgba(35, 38, 47, 0.92) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-left: 0 !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.18) !important;
    }

    header[data-testid="stHeader"] button:has(svg) svg {
        width: 0.9rem !important;
        height: 0.9rem !important;
    }

    /* ── Chat message images ─────────────────────────────── */
    .chat-thumb {
        border-radius: 12px;
        max-height: 180px;
        margin: 4px 4px 8px 0;
        border: 1px solid rgba(128,128,128,0.25);
    }

    /* ── Title bar ───────────────────────────────────────── */
    .title-bar {
        text-align: center;
        padding: 0.3rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        margin-bottom: 1rem;
    }

    .title-bar h1 {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .title-bar p {
        font-size: 0.8rem;
        opacity: 0.55;
        margin: 0.15rem 0 0 0;
    }

    /* ── Heatmap overlay images ───────────────────────────── */
    .gradcam-img {
        border-radius: 10px;
        max-width: 340px;
        margin: 8px 0;
        border: 1px solid rgba(255,100,100,0.3);
    }

    /* ── Responsive ──────────────────────────────────────── */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            max-width: 100%;
        }
        .chat-thumb {
            max-height: 120px;
        }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PIPELINE LOADING (cached — runs once)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading T-RAG pipeline — this may take a minute…")
def _sidebar_toggle_cache_anchor():
    return None


@st.cache_resource(show_spinner="Loading T-RAG pipeline - this may take a minute...")
def load_pipeline():
    from src.chatbot.pipeline import ChatPipeline
    pipe = ChatPipeline()
    pipe.load()
    return pipe


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

if "uploaded_meta" not in st.session_state:
    st.session_state.uploaded_meta = None

if "image_temp_paths" not in st.session_state:
    st.session_state.image_temp_paths = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = time.time()

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


def _chat_title(messages: list) -> str:
    """Use the first user message as a compact chat title."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            title = " ".join(msg["content"].split())
            return title[:48] + ("..." if len(title) > 48 else "")
    return "Untitled chat"


def _save_current_chat() -> None:
    """Save or update the active chat in session history."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return

    chat_id = st.session_state.current_chat_id
    session = {
        "id": chat_id,
        "title": _chat_title(messages),
        "messages": list(messages),
        "updated_at": time.time(),
    }

    sessions = [
        s for s in st.session_state.chat_sessions
        if s.get("id") != chat_id
    ]
    st.session_state.chat_sessions = [session] + sessions[:19]


def _start_new_chat() -> None:
    _save_current_chat()
    st.session_state.messages = []
    st.session_state.uploaded_images = []
    st.session_state.uploaded_meta = None
    st.session_state.image_temp_paths = []
    st.session_state.current_chat_id = time.time()


def _load_chat(session: dict) -> None:
    st.session_state.messages = list(session.get("messages", []))
    st.session_state.current_chat_id = session.get("id", time.time())


def _delete_chat(chat_id) -> None:
    st.session_state.chat_sessions = [
        s for s in st.session_state.chat_sessions
        if s.get("id") != chat_id
    ]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = time.time()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### TM2-RAG")
    st.caption("An Automated Framework for Medical Report Generation")

    st.divider()

    if st.button("New Chat", use_container_width=True):
        _start_new_chat()
        st.rerun()

    chat_search = st.text_input(
        "Search Chat",
        key="chat_search",
        placeholder="Search recent chats",
    )

    st.markdown("**Recent Chats**")
    recent_chats = st.session_state.chat_sessions
    if chat_search:
        needle = chat_search.lower()
        recent_chats = [
            s for s in recent_chats
            if needle in s.get("title", "").lower()
            or any(
                needle in msg.get("content", "").lower()
                for msg in s.get("messages", [])
            )
        ]

    if recent_chats:
        for session in recent_chats[:10]:
            chat_col, delete_col = st.columns([0.78, 0.22])
            with chat_col:
                if st.button(
                    session.get("title", "Untitled chat"),
                    key=f"chat_{session.get('id')}",
                    use_container_width=True,
                    disabled=st.session_state.is_processing,
                ):
                    _save_current_chat()
                    _load_chat(session)
                    st.rerun()
            with delete_col:
                if st.button(
                    "Delete",
                    key=f"delete_chat_{session.get('id')}",
                    use_container_width=True,
                    disabled=st.session_state.is_processing,
                ):
                    _delete_chat(session.get("id"))
                    st.rerun()
    else:
        st.caption("No recent chats yet.")

    st.divider()

    st.markdown(
        "**Model:** Qwen2.5-VL-3B  \n"
        "**Quantization:** 4-bit NF4  \n"
        "**Retrieval:** Reports + Windows + Trajectories"
    )

    st.divider()
    st.markdown("**Open App**")
    st.code("http://localhost:8501", language=None)
    st.caption("Other devices on the same local network can try:")
    st.code(get_lan_url(), language=None)
    st.caption("For a different location, use a tunnel or VPN; the Network URL is LAN-only.")

    st.divider()
    st.caption(
        "Upload chest X-ray images, corresponding reports, and/or patient metadata "
        "along with your query. Ask for Grad-CAM heatmaps by "
        "including words like *heatmap*, *highlight*, or *grad-cam*."
    )

    st.checkbox(
        "Generate Grad-CAM for uploaded images",
        key="force_gradcam",
        value=False,
    )


# ═══════════════════════════════════════════════════════════════
# TITLE BAR
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<div class="title-bar">'
    '<h1> TM2-RAG - Your AI Radiology Assistant</h1>'
    '<p>An Automated Framework for Medical Report Generation · Qwen2.5-VL</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# FILE UPLOADERS (above the chat input, like ChatGPT)
# ═══════════════════════════════════════════════════════════════

upload_col1, upload_col2, upload_col3 = st.columns(3)

with upload_col1:
    image_files = st.file_uploader(
        "Chest X-ray image(s)",
        type=["jpg", "jpeg", "png", "dcm"],
        accept_multiple_files=True,
        key="img_uploader",
        label_visibility="visible",
        help="Upload chest X-ray images (AP / Lateral)",
    )
    st.caption("Supported: JPG, JPEG, PNG, DCM. Upload one or more chest radiographs.")

with upload_col2:
    report_files = st.file_uploader(
        "Corresponding report file(s)",
        type=["txt", "csv", "json"],
        accept_multiple_files=True,
        key="report_uploader",
        label_visibility="visible",
        help="Upload prior/current radiology reports. They are matched to images by upload order or filename timestamps.",
    )
    st.caption("Supported: TXT, CSV, JSON. Upload reports for the same studies/timepoints.")

with upload_col3:
    meta_file = st.file_uploader(
        "Patient metadata (CSV / JSON / TXT)",
        type=["csv", "json", "txt"],
        accept_multiple_files=False,
        key="meta_uploader",
        label_visibility="visible",
        help="Upload patient metadata file",
    )
    st.caption("Supported: CSV, JSON, TXT. Optional demographics or study metadata.")

if image_files and report_files and len(image_files) != len(report_files):
    st.caption(
        "Note: uploaded image/report counts differ. Reports will still be used as temporal context; matching is by upload order or filename dates/study IDs."
    )


# ═══════════════════════════════════════════════════════════════
# PROCESS UPLOADS → save images to temp, parse metadata
# ═══════════════════════════════════════════════════════════════

def save_uploaded_images(files) -> list:
    """Save uploaded image files to temp dir and return paths."""
    paths = []
    for f in files:
        suffix = Path(f.name).suffix
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="trag_img_"
        )
        tmp.write(f.getvalue())
        tmp.close()
        paths.append(tmp.name)
    return paths


def parse_uploaded_meta(f) -> str:
    """Parse uploaded metadata file into text."""
    if f is None:
        return ""
    from src.chatbot.pipeline import parse_metadata_file
    return parse_metadata_file(f.getvalue(), f.name)


def _infer_study_label(filename: str, index: int) -> str:
    """Infer a readable study label from a report/image filename."""
    stem = Path(filename).stem
    date_match = re.search(r"(20\d{2}|19\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    if date_match:
        yyyy, mm, dd = date_match.groups()
        return f"{yyyy}-{mm}-{dd}"
    clean_name = re.sub(r"[_-]+", " ", stem).strip()
    return clean_name or f"Study {index + 1}"


def parse_uploaded_reports(files) -> str:
    """Combine uploaded report files into temporal context text."""
    if not files:
        return ""

    lines = [
        f"Uploaded radiology reports ({len(files)} studies):",
        "Reports are paired with uploaded X-ray images by upload order unless filenames contain matching dates or study IDs.",
    ]

    for i, f in enumerate(files[:10]):
        text = f.getvalue().decode("utf-8", errors="replace").strip()
        label = _infer_study_label(f.name, i)
        lines.append(
            f"\nStudy {i + 1}: {label}\n"
            f"Report file: {f.name}\n"
            f"{text[:2500]}"
        )

    if len(files) > 10:
        lines.append(f"\n... and {len(files) - 10} more uploaded report file(s).")

    return "\n".join(lines)


def build_uploaded_context(metadata_text: str, report_text: str) -> str:
    """Merge optional metadata and uploaded report timeline into one prompt block."""
    parts = []
    if metadata_text:
        parts.append("PATIENT METADATA:\n" + metadata_text)
    if report_text:
        parts.append("UPLOADED TEMPORAL REPORT TIMELINE:\n" + report_text)
    return "\n\n".join(parts)


def render_heatmap(hm):
    """Render a Grad-CAM image with label/confidence when available."""
    if isinstance(hm, dict):
        caption = (
            f"{hm.get('label', 'Grad-CAM')} "
            f"({hm.get('confidence', 0.0):.2f})"
        )
        st.image(hm.get("image"), caption=caption, width=320)
    else:
        st.image(hm, caption="Grad-CAM Overlay", width=320)


# ═══════════════════════════════════════════════════════════════
# RENDER CHAT HISTORY
# ═══════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show image thumbnails in user messages
        if msg["role"] == "user" and msg.get("images"):
            cols = st.columns(min(len(msg["images"]), 3))
            for i, img_path in enumerate(msg["images"]):
                with cols[i % 3]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, width=180)
                    except Exception:
                        st.caption(f"📷 {Path(img_path).name}")

        if msg["role"] == "user" and msg.get("reports"):
            st.caption("Uploaded reports: " + ", ".join(msg["reports"]))

        st.markdown(msg["content"])

        # Show Grad-CAM overlays
        if msg.get("heatmaps"):
            st.markdown("**🔥 Grad-CAM Heatmap(s):**")
            hm_cols = st.columns(min(len(msg["heatmaps"]), 3))
            for i, hm in enumerate(msg["heatmaps"]):
                with hm_cols[i % 3]:
                    render_heatmap(hm)


# ═══════════════════════════════════════════════════════════════
# CHAT INPUT
# ═══════════════════════════════════════════════════════════════

if st.session_state.is_processing:
    st.info("A response is currently being generated. Please wait until it finishes before sending another message.")

user_input = st.chat_input(
    "Ask about a chest X-ray, describe findings, or request a report…"
    , disabled=st.session_state.is_processing,
)

if user_input and not st.session_state.is_processing:
    st.session_state.pending_user_input = user_input
    st.session_state.is_processing = True
    st.rerun()

active_user_input = st.session_state.pop("pending_user_input", None)

if active_user_input:
    user_input = active_user_input

    # ── Save uploaded files ──────────────────────────────────
    image_paths = save_uploaded_images(image_files) if image_files else []
    uploaded_report_text = parse_uploaded_reports(report_files)
    metadata_text = build_uploaded_context(
        parse_uploaded_meta(meta_file),
        uploaded_report_text,
    )
    report_names = [f.name for f in report_files] if report_files else []

    # ── Add user message to history ──────────────────────────
    user_msg = {
        "role": "user",
        "content": user_input,
        "images": image_paths,
        "reports": report_names,
    }
    st.session_state.messages.append(user_msg)

    # ── Render user message ──────────────────────────────────
    with st.chat_message("user"):
        if image_paths:
            cols = st.columns(min(len(image_paths), 3))
            for i, img_path in enumerate(image_paths):
                with cols[i % 3]:
                    try:
                        st.image(Image.open(img_path), width=180)
                    except Exception:
                        st.caption(f"📷 {Path(img_path).name}")
        if report_names:
            st.caption("Uploaded reports: " + ", ".join(report_names))
        st.markdown(user_input)

    # ── Generate response ────────────────────────────────────
    with st.chat_message("assistant"):
        # Load pipeline
        pipeline = load_pipeline()

        # Check if user wants heatmaps
        from src.chatbot.pipeline import wants_heatmap, wants_heatmap_only
        heatmap_only = wants_heatmap_only(user_input)
        do_heatmap = (
            (wants_heatmap(user_input) or st.session_state.get("force_gradcam", False))
            and len(image_paths) > 0
        )
        print(
            f"[App] Grad-CAM requested={do_heatmap} "
            f"images={len(image_paths)} "
            f"force={st.session_state.get('force_gradcam', False)} "
            f"only={heatmap_only}"
        )

        # Stream the response
        response_placeholder = st.empty()
        full_response = ""
        work_status = st.status("Preparing your request...", expanded=False)

        if heatmap_only and do_heatmap:
            full_response = "Grad-CAM heatmap generated for the uploaded image."
            response_placeholder.markdown(full_response)
            work_status.update(
                label="Skipping report generation; preparing Grad-CAM...",
                state="running",
            )
        else:
            try:
                work_status.update(
                    label="Retrieving temporal context and loading the model...",
                    state="running",
                )
                first_token = True
                for token in pipeline.generate_stream(
                    user_query=user_input,
                    image_paths=image_paths if image_paths else None,
                    metadata_text=metadata_text,
                ):
                    if first_token:
                        work_status.update(
                            label="Generating report...",
                            state="running",
                        )
                        first_token = False
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                work_status.update(label="Response complete.", state="complete")

            except Exception as e:
                full_response = f"⚠️ Generation error: {str(e)}"
                response_placeholder.markdown(full_response)
                work_status.update(label="Generation failed.", state="error")

        # ── Grad-CAM (if requested) ──────────────────────────
        heatmap_images = []
        if do_heatmap:
            with st.spinner("Generating Grad-CAM heatmaps…"):
                try:
                    work_status.update(
                        label="Generating Grad-CAM overlay...",
                        state="running",
                    )
                    heatmap_images = pipeline.generate_heatmaps(
                        image_paths=image_paths,
                        report_text=full_response,
                    )
                    if heatmap_images:
                        st.markdown("**🔥 Grad-CAM Heatmap(s):**")
                        hm_cols = st.columns(min(len(heatmap_images), 3))
                        for i, hm in enumerate(heatmap_images):
                            with hm_cols[i % 3]:
                                render_heatmap(hm)
                    else:
                        st.warning("Grad-CAM did not produce any overlays.")
                    work_status.update(label="Grad-CAM complete.", state="complete")
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")
                    work_status.update(label="Grad-CAM failed.", state="error")

        # ── Save assistant message ───────────────────────────
        assistant_msg = {
            "role": "assistant",
            "content": full_response,
        }
        if heatmap_images:
            assistant_msg["heatmaps"] = heatmap_images

        st.session_state.messages.append(assistant_msg)
        _save_current_chat()
        st.session_state.is_processing = False
        st.rerun()
