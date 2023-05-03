# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import utils
import intro_app
import data_app
import eda_app
import stat_app

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu("Main Menu", ["INTRO", "DATA", "EDA", "STAT", "ML"],
                               icons=["house", "card-checklist", "bar-chart", "clipboard-data", "gear"],
                               menu_icon="cast",
                               default_index=0,
                               orientation="vertical")
    st.title("Store Sales")

    if selected == "INTRO":
        intro_app.intro_app()
    if selected == "DATA":
        data_app.data_app()
    if selected == "EDA":
        eda_app.eda_app()
    if selected == "STAT":
        st.subheader("STATS")
    if selected == "ML":
        st.subheader("Machine Learning")

if __name__ == "__main__":
    main()