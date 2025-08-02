import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import config.conf as conf
load_dotenv()


def clear_messages():
    st.session_state.analysis_messages = {"image": [],
                                          "analysis" : [],
                                          "diagnosis" : []
                                          }
    st.session_state.disease_detected = False

st.set_page_config(
    page_title = "Harvest Analytics",
    page_icon = "👨🏻‍🌾",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Set Title
title_col = st.columns([0.8, 12, 0.5])
with title_col[0]:
    st.image(image = Path(conf.ASSETS_DIR, "harvest.png"), use_container_width = True)
with title_col[1]:
    st.header("Harvest Analytics", divider = "rainbow", anchor = False)
with title_col[2]:
    st.write("")
    st.button(":material/mop:", on_click = lambda: clear_messages(), type = "secondary", use_container_width = True)

