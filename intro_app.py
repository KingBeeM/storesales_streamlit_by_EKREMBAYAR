# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import utils

def intro_app():
    tab1, tab2 = st.tabs(["Introduction", "Misson"])
    with tab1:
        st.subheader("Introduction")

        st.write("에콰도르의 **Corporación Favorita** 라는 대형 식료품 소매 업체의 데이터 입니다.")

        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")
        with col2:
            st.image(utils.e_img1, width = 250)
            st.image(utils.e_img2, width = 250)
        with col3:
            st.write("")
        st.write("Corporación Favorita 은 남미의 다른 국가에서도 사업을 운영하고 있습니다.")
        st.write("우리는 **54개의 Corporación Favorita 의 지점**과 **33개의 제품**에 관한 데이터를 통해 매출 예상을 할 예정입니다.")
        st.write("우리가 가지고 있는 기간은 **2013-01-01 ~ 2017-08-31** 입니다.")

    with tab2:
        st.markdown("## Goal of the Competition \n"
                    "- In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer. \n"
                    "- Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales. \n")

        st.markdown("## Evaluation \n"
                    "- The evaluation metric for this competition is Root Mean Squared Logarithmic Error. \n")
        st.latex(r'''
        {RMSLE} = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n}}
        ''')
        st.markdown("where: \n"
                    "- $n$ is the total number of instances \n"
                    "- $\hat{y}_i$ is the predicted value of the target for instance (i) \n"
                    "- $y_i$ is the actual value of the target for instance (i), and, \n"
                    "- $\log$ is the natural logarithm \n"
                    )

        st.markdown("### Competition Info \n"
                    "More Detailed : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)")
