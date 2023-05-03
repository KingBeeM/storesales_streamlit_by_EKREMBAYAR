# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from PIL import Image

# image url
e_img1 = "https://www.corporacionfavorita.com/wp-content/uploads/2020/03/logo-cf-footer.png"
e_img2 = "https://www.corporacionfavorita.com/wp-content/uploads/2020/05/cf-donde_estamos-2020.jpg"


# data path
holidays_path = "data/holidays_events.csv"
oil_path = "data/oil.csv"
stores_path = "data/stores.csv"
test_path = "data/test.csv"
train_path = "data/train.csv"
transactions_path = "data/transactions.csv"

@st.cache_data
def load_data():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    transactions = pd.read_csv(transactions_path)
    stores = pd.read_csv(stores_path)
    oil = pd.read_csv(oil_path)
    holidays = pd.read_csv(holidays_path)

    return train, test, transactions, stores, oil, holidays