import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Backend Assistant", layout="wide")

st.title("🤖 AI Backend Assistant")

# -------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- FILE UPLOAD --------
st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}

    with st.spinner("Uploading..."):
        res = requests.post(f"{API_URL}/upload", files=files)

    if res.status_code == 200:
        st.sidebar.success("Uploaded successfully!")
    else:
        st.sidebar.error("Upload failed")

# -------- CHAT DISPLAY --------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- USER INPUT --------
prompt = st.chat_input("Ask something...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": prompt}
        )

    if response.status_code == 200:
        data = response.json()

        if data.get("success"):
            answer = data["data"]["answer"]

            # Format details nicely
            details = data["data"].get("details", [])

            details_text = ""
            if details:
                details_text = "\n\n**Details:**\n"
                for d in details:
                    details_text += f"- {d}\n"

            full_response = answer + details_text

        else:
            full_response = data.get("error", "Error occurred")

    else:
        full_response = "API error"

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })