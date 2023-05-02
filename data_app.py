# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import utils


@st.cache_data
def load_data():
    train = pd.read_csv(utils.train_path)
    test = pd.read_csv(utils.test_path)
    transactions = pd.read_csv(utils.transactions_path)
    stores = pd.read_csv(utils.stores_path)
    oil = pd.read_csv(utils.oil_path)
    holidays = pd.read_csv(utils.holidays_path)

    return train, test, transactions, stores, oil, holidays

def data_app():
    train, test, transactions, stores, oil, holidays = load_data()
    datalist = st.sidebar.selectbox("SELECT DATA",["Train", "Test", "Transactions", "Stores", "Oil", "Holidays_Events"])

    st.subheader(f"{datalist} DATA")
    if datalist == "Train":
        st.write(train.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(train.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(train.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(train["store_nbr"].value_counts())
            with col2:
                st.write(train["family"].value_counts())

    if datalist == "Test":
        st.write(test.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(test.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(test.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(test["store_nbr"].value_counts())
            with col2:
                st.write(test["family"].value_counts())

    if datalist == "Transactions":
        st.write(transactions.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(transactions.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(transactions.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(transactions["store_nbr"].value_counts())
            with col2:
                pass

    if datalist == "Stores":
        st.write(stores.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(stores.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(stores.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(stores["store_nbr"].value_counts())
            with col2:
                pass

    if datalist == "Oil":
        st.write(oil.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(oil.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(oil.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(oil["store_nbr"].value_counts())
            with col2:
                pass

    if datalist == "Holidays_Events":
        st.write(holidays.head(10))
        st.write(f"{datalist} 데이터 입니다. \n 컬럼 설명")
        with st.expander(f"{datalist}.dtypes"):
            st.write(holidays.dtypes)
        with st.expander(f"{datalist}.describe()"):
            st.write(holidays.describe())
        with st.expander(f"{datalist}.value_counts()"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(holidays["store_nbr"].value_counts())
            with col2:
                pass
