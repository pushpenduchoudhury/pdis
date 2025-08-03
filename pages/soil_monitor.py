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
    page_title = "Soil Monitor",
    page_icon = "⛰️",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Set Title
title_col = st.columns([0.8, 12, 0.5])
with title_col[0]:
    st.image(image = Path(conf.ASSETS_DIR, "soil.png"), use_container_width = True)
with title_col[1]:
    st.header("Soil Monitoring System", divider = "rainbow", anchor = False)
with title_col[2]:
    st.write("")
    st.button(":material/mop:", on_click = lambda: clear_messages(), type = "secondary", use_container_width = True)

col1, col2 = st.columns([0.07, 0.8])
col1.image(Path(conf.ASSETS_DIR, "ideation.png"), use_container_width = True)
col2.markdown("""#### :grey[Soil Monitor is a precision agriculture app providing real-time insights into soil health. Using sensor data and predictive analytics, it helps farmers optimize irrigation, fertilization, and planting strategies, leading to improved crop yields and reduced input costs. Features include detailed soil moisture maps, nutrient level analysis, and customized recommendations based on specific field conditions and weather forecasts.]""")

