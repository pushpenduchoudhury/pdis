import math
import streamlit as st
from pathlib import Path
import config.conf as conf
from dotenv import load_dotenv
load_dotenv()

CSS_FILE = Path(conf.CONFIG_DIR, "style.css")

with open(CSS_FILE) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html = True)

st.header("Agri AI", anchor = False, divider = "red")

apps = [
    {"name": "PDIS",
     "description": "The Plant Disease Identification System (PDIS) is an AI-powered tool designed to detect and diagnose plant diseases from images of leaves, stems, or fruits. Using advanced image processing and machine learning models, PDIS helps farmers, gardeners, and researchers quickly identify diseases, recommend preventive measures, and suggest appropriate treatments, enabling early intervention and improved crop health.",
     "page": "pdis.py",
     "image_icon": "logo.png",
    },
]

no_of_apps = len(apps)
app_grid_cols = 4
app_grid_rows = math.ceil(no_of_apps/app_grid_cols)
tile_height = 400
image_width = 65


app_num = 0
for row in range(app_grid_rows):
    st_cols = st.columns(app_grid_cols)
    for col in range(app_grid_cols):
        if app_num > no_of_apps - 1:
            break
        with st_cols[col].container(border = True, height = tile_height):
            
            # Image
            st.image(image = str(Path(conf.ASSETS_DIR, apps[app_num]["image_icon"])), width = image_width)
            
            # App Title
            st.subheader(apps[app_num]["name"], divider = "grey", anchor = False)
            
            # App Description
            desc_col = st.columns(1)
            with desc_col[0].container(border = False, height = int(0.40 * tile_height)):
                st.markdown(f'<span style="font-size: 16px; text-align: center;">{apps[app_num]["description"]}</span>', unsafe_allow_html = True)
            
            # App Launch Button
            if st.button("Launch", key = f"app_{app_num}"):
                st.switch_page(Path(conf.PAGES_DIR, apps[app_num]["page"]))
                
            app_num += 1