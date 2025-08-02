import uuid
import streamlit as st
from pathlib import Path
import config.conf as conf
from dotenv import load_dotenv
load_dotenv()


# Set Page Config
st.set_page_config(
    page_title = "GenAI Hub",
    page_icon = "🤖",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Initialize chat session in streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
st.markdown(
    f"""
<style>
    .st-emotion-cache-1xgtwnd:before {{
        content: "𖡎 PDIS";
        font-weight: bold;
        font-size: xx-large;
    }}
</style>""",
        unsafe_allow_html=True,
    )


pages = {
        "🏠︎ Home": [st.Page(Path(conf.PAGES_DIR, "home.py"), title = "Homepage", icon = ":material/home:", default = True)],
        "📰 Apps" : [st.Page(Path(conf.PAGES_DIR, "pdis.py"), title = "Plant Disease Identification", icon = ":material/coronavirus:")],
        "❔ Help" : [st.Page(Path(conf.PAGES_DIR, "about.py"), title = "About", icon = ":material/info:")],
}

page = st.navigation(pages, position = "top")
page.run()