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


col1, col2 = st.columns([0.07, 0.8])
col1.image(Path(conf.ASSETS_DIR, "ideation.png"), use_container_width = True)
col2.markdown("""#### :grey[Harvest Analytics empowers farmers with data-driven decision-making. Leveraging advanced predictive analytics, we forecast crop yields by integrating real-time weather data, soil conditions, and historical harvest information. Our detailed reports provide insights into optimal harvesting times, potential yield variations, and resource allocation strategies. Track your harvest progress, analyze key performance indicators, and access historical data for continuous improvement and enhanced profitability.]""")

