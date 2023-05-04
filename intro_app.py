# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import utils

def intro_app():
    # st.markdown(
    #     """
    #     <style>
    #     /* 제목의 색상과 폰트 변경 */
    #     h3 {
    #         color: skyblue;
    #         font-family: "Helvetica", sans-serif;
    #         font-size: 28px;
    #         text-align: left;
    #     }
    #     h1 {
    #         color: black;
    #         font-family: "Helvetica", sans-serif;
    #         font-size: 80px;
    #         text-align: center;
    #     /* 텍스트 블록을 오른쪽으로 정렬 */
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    tab1, tab2 = st.tabs(["**소개**", "**목표**"])
    with tab1:
        st.subheader("소개")
        st.markdown(
            "<span style='font-size: 24px;'>에콰도르의 <b>Corporación Favorita</b> 라는 대형 식료품 소매 업체의 데이터 입니다.</span>",
            unsafe_allow_html=True)
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([2, 6, 1])
        with col1:
            st.write("")
        with col2:
            st.image(utils.e_img1, width=280)
            st.image(utils.e_img2, width=280)
        with col3:
            st.write("")

        st.write("")
        st.write("")
        st.write(
            "**Supermaxi** 대형 유통 체인을 운영하고 있는 기업으로 잘 알려진 **Corporación Favorita**는 에콰도르에서 활동하고 있는 기업들 중 **매출액 1위**를 유지하고 있는 유망한 기업입니다.")
        st.write(
            "**Corporación Favorita** 은 남미의 다른 국가에서도 사업을 운영하고 있으며 총 **54개의 Corporación Favorita 의 지점**과 **33개의 제품**에 관한 데이터를 통해 앞으로의 매출액을 예측할 예정입니다.")
        st.write("그리고 데이터 분석을 위해 제공된 **Corporación Favorita**의 데이터는 **2013-01-01 ~ 2017-08-31** 까지의 데이터입니다.")

    with tab2:
        st.markdown('### 대회 목표 \n'
                    '- 이번 대회의 목표는 시계열 예측을 사용하여 에콰도르에 본사를 두고 있는 대형 식료품 소매업체인 \"Corporación Favorita\"의 데이터를 분석하고 매장의 미래 매출을 예측하는 것입니다.\n'
                    '- 구체적으로는 여러 Favorita 매장에서 판매되는 수많은 품목의 판매 단가를 보다 정확하게 예측하는 모델을 구축하는 것이 최종 목표입니다.\n'
                    '- 날짜, 매장 및 품목 정보, 프로모션, 판매 단가로 구성된 접근하기 쉬운 학습 데이터 셋을 통해 ML 기술들을 연습할 수도 있습니다.\n')
        st.markdown("---")
        st.markdown("### 평가 \n"
                    "- 이 대회의 평가 지표는 Root Mean Squared Logarithmic Error입니다. \n")
        st.latex(r'''
            {RMSLE} = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n}}
            ''')
        st.markdown(
                    "- $n$ n은 총 인스턴스의 수입니다. \n"
                    "- $\hat{y}_i$ i는 인스턴스 i에 대한 예측된 타겟 값입니다. \n"
                    "- $y_i$ 는 인스턴스 i에 대한 실제 타겟입니다. \n"
                    "- $\log$ 는 자연 로그입니다. \n"
                    )
        st.markdown("---")
        st.markdown("### 대회 정보 \n"
                    "More Detailed : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)")

    # with tab3:
    #     st.markdown("<h2><center>File Description and Data Field Information</center></h2>", unsafe_allow_html=True)
    #
    #     st.write("\n\n")
    #
    #     st.markdown("<h3>📝train.csv</h4>", unsafe_allow_html=True)
    #     st.write(
    #         '- The training data, comprising time series of features <b>store_nbr</b>, <b>family</b>, and <b>onpromotion</b> as well as the target <b>sales</b>.',
    #         unsafe_allow_html=True)
    #     st.write('- <b>store_nbr</b> identifies the store at which the products are sold.', unsafe_allow_html=True)
    #     st.write('- <b>family</b> identifies the type of product sold.', unsafe_allow_html=True)
    #     st.write(
    #         '- <b>sales</b> gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).',
    #         unsafe_allow_html=True)
    #     st.write(
    #         '- <b>onpromotion</b> gives the total number of items in a product family that were being promoted at a store at a given date.',
    #         unsafe_allow_html=True)
    #
    #     st.write("\n\n")
    #
    #     st.markdown("<h3>📝test.csv</h3>", unsafe_allow_html=True)
    #     st.write(
    #         '- The test data, having the same features as the training data. You will predict the target <b>sales</b> for the dates in this file.',
    #         unsafe_allow_html=True)
    #     st.write('- The dates in the test data are for the 15 days after the last date in the training data.')
    #
    #     st.write("\n\n")
    #     st.markdown("<h3>📝sample submission.csv</h3>", unsafe_allow_html=True)
    #     st.write("- A sample submission file in the correct format.")
    #
    #     st.write("\n\n")
    #     st.markdown("<h3>📝stores.csv</h3>", unsafe_allow_html=True)
    #     st.write('- Store metadata, including <b>city, state, type</b>, and <b>cluster</b>.', unsafe_allow_html=True)
    #     st.write('- <b>cluster</b> is a grouping of similar stores.', unsafe_allow_html=True)
    #
    #     st.write("\n\n")
    #     st.markdown("<h3>📝oil.csv</h3>", unsafe_allow_html=True)
    #     st.write(
    #         "- Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)")
    #
    #     st.write("\n\n")
    #     st.markdown("<h3>📝holidays_events.csv</h3>", unsafe_allow_html=True)
    #     st.write('- Holidays and Events, with metadata')
    #     st.write('- NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. \
    #         To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was <b>transferred</b> from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. \
    #         Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). \
    #         These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.')
    #     st.write(
    #         '- Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).')
    #
    #     st.write("\n\n")
    #     st.markdown("<h3>📝Additional Note.csv</h3>", unsafe_allow_html=True)
    #     st.write(
    #         '- Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.')
    #     st.write(
    #         '- A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.')

    # tab1, tab2 = st.tabs(["Introduction", "Misson"])
    # with tab1:
    #     st.subheader("Introduction")
    #
    #     st.write("에콰도르의 **Corporación Favorita** 라는 대형 식료품 소매 업체의 데이터 입니다.")
    #
    #     col1, col2, col3 = st.columns([1, 6, 1])
    #     with col1:
    #         st.write("")
    #     with col2:
    #         st.image(utils.e_img1, width = 250)
    #         st.image(utils.e_img2, width = 250)
    #     with col3:
    #         st.write("")
    #     st.write("Corporación Favorita 은 남미의 다른 국가에서도 사업을 운영하고 있습니다.")
    #     st.write("우리는 **54개의 Corporación Favorita 의 지점**과 **33개의 제품**에 관한 데이터를 통해 매출 예상을 할 예정입니다.")
    #     st.write("우리가 가지고 있는 기간은 **2013-01-01 ~ 2017-08-31** 입니다.")
    #
    # with tab2:
    #     st.markdown("## Goal of the Competition \n"
    #                 "- In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer. \n"
    #                 "- Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales. \n")
    #
    #     st.markdown("## Evaluation \n"
    #                 "- The evaluation metric for this competition is Root Mean Squared Logarithmic Error. \n")
    #     st.latex(r'''
    #     {RMSLE} = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n}}
    #     ''')
    #     st.markdown("where: \n"
    #                 "- $n$ is the total number of instances \n"
    #                 "- $\hat{y}_i$ is the predicted value of the target for instance (i) \n"
    #                 "- $y_i$ is the actual value of the target for instance (i), and, \n"
    #                 "- $\log$ is the natural logarithm \n"
    #                 )
    #
    #     st.markdown("### Competition Info \n"
    #                 "More Detailed : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)")
