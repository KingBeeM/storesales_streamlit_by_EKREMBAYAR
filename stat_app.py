# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import utils
import eda_app

@st.cache_data
def eda_features_date(train, test, transactions, stores, oil, holidays):
    """
    eda_app 에서 데이터 전처리 한 과정을 반환하는 함수

    :param train:
    :param test:
    :param transactions:
    :param stores:
    :param oil:
    :param holidays:
    :return: train, test, transactions, stores, oil, holidays, eda_app.Feature_Engineering_Holidays(holidays, train, test, stores)
    """

    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)
    holidays["date"] = pd.to_datetime(holidays.date)

    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()

    train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
    train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
    train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
    train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
    train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
    train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
    train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
    train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]

    c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family", "store_nbr"])
    c = c[c.sales == 0]

    outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how="outer", indicator=True)
    train = outer_join[~(outer_join._merge == "both")].drop("_merge", axis=1)

    d = eda_app.Feature_Engineering_Holidays(holidays, train, test, stores)

    return train, test, transactions, stores, oil, holidays, d

def create_date_features(df):
    """
    date 정보를 여러 개로 나눠서 패턴을 파악하기 위한 데이터 피쳐
    """
    df["month"] = df.date.dt.month.astype("int8")
    df["day_of_month"] = df.date.dt.day.astype("int8")
    df["day_of_year"] = df.date.dt.dayofyear.astype("int16")
    df["week_of_month"] = (df.date.apply(lambda d: (d.day-1)//7 + 1)).astype("int8")
    df["week_of_year"] = df.date.dt.weekofyear.astype("int8")
    df["day_of_week"] = (df.date.dt.dayofweek + 1).astype("int8")
    df["year"] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday//4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df["is_month_start"] = df.date.dt.is_month_start.astype("int8")
    df["is_month_end"] = df.date.dt.is_month_end.astype("int8")
    df["is_quarter_start"] = df.date.dt.is_quarter_start.astype("int8")
    df["is_quarter_end"] = df.date.dt.is_quarter_end.astype("int8")
    df["is_year_start"] = df.date.dt.is_year_start.astype("int8")
    df["is_year_end"] = df.date.dt.is_year_end.astype("int8")

    # 0 : Winter , 1 : Spring , 2 : Summer , 3 : Fall
    df["season"] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df["season"] = np.where(df.month.isin([6, 7, 8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")

    return df

def ewm_features(dataframe, alphas, lags):
    """
    지수 평균 이동 반환 함수
    """
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe["sales_ewm_alpha_" + str(alpha).replace(".","") + "_lag_" + str(lag)] = \
            dataframe.groupby(["store_nbr", "family"])["sales"].\
            transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

def fig_acf_pacf(a):
    """
    제품별 (family)에 대한 Sales 정보를 ACF 와 PACF 상관관계에 따라 보여주는 함수
    lags 값은 365를 사용하였으나 데이터 용량 문제로 자르면서 임시로 363 값을 줌
    """
    for num, i in enumerate(a.family.unique()):
       try:
           fig, ax = plt.subplots(1, 2, figsize=(15, 5))
           temp = a[(a.family == i)]
           sm.graphics.tsa.plot_acf(temp.sales, lags=363, ax=ax[0], title="Auto Correlation\n" + i)
           sm.graphics.tsa.plot_pacf(temp.sales, lags=363, ax=ax[1], title="Partial Auto Correlation\n" + i)
           st.pyplot(fig)
       except:
           pass
def fig_average_sales(a):
    """
    연도에 따라 판매평균을 비교하기 위한 그래프
    파라메터 a 값에 따라 보여주는 연도가 달라진다.
    """
    fig, ax = plt.subplots()
    fig = px.line(a, x="day_of_year", y="sales", color="year", title=f"Average Sales for {a.year.unique()[0]} and {a.year.unique()[1]}")
    st.plotly_chart(fig)

def fig_SMA_graph(a):
    """
    단순이동평균을 보여주는 그래프
    """
    for i in a.family.unique():
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))

        a[a.family == i][["sales", "SMA20_sales_lag16"]].plot(legend=True, ax=ax[0, 0], linewidth=4)
        a[a.family == i][["sales", "SMA30_sales_lag16"]].plot(legend=True, ax=ax[0, 1], linewidth=4)
        a[a.family == i][["sales", "SMA45_sales_lag16"]].plot(legend=True, ax=ax[0, 2], linewidth=4)
        a[a.family == i][["sales", "SMA60_sales_lag16"]].plot(legend=True, ax=ax[0, 3], linewidth=4)
        a[a.family == i][["sales", "SMA90_sales_lag16"]].plot(legend=True, ax=ax[1, 0], linewidth=4)
        a[a.family == i][["sales", "SMA120_sales_lag16"]].plot(legend=True, ax=ax[1, 1], linewidth=4)
        a[a.family == i][["sales", "SMA365_sales_lag16"]].plot(legend=True, ax=ax[1, 2], linewidth=4)
        a[a.family == i][["sales", "SMA730_sales_lag16"]].plot(legend=True, ax=ax[1, 3], linewidth=4)

        plt.suptitle("STORE 1 - " + i, fontsize=15)
        plt.tight_layout(pad=1.5)
        for j in range(0, 4):
            ax[0, j].legend(fontsize="x-large")
            ax[1, j].legend(fontsize="x-large")

        st.pyplot(fig)

def fig_EMA_graph(a):
    """
    지수평균이동을 보여주는 그래프
    """
    fig, ax = plt.subplots()
    a[(a.store_nbr == 1) & (a.family == "GROCERY I")].set_index("date")[["sales", "sales_ewm_alpha_095_lag_16"]].plot(title=f"STORE 1 - GROCERY I", ax=ax)
    st.pyplot(fig)

def stat_app():
    # load_data & features_data
    train, test, transactions, stores, oil, holidays = utils.load_data()
    train, test, transactions, stores, oil, holidays, d = eda_features_date(train, test, transactions, stores, oil, holidays)

    # Time Related Features
    d = create_date_features(d)

    # Workday column
    d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary == 1) | (d.holiday_regional_binary == 1) | (d['day_of_week'].isin([6, 7])), 0, 1)
    d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
    d.drop("IsWorkDay", axis=1, inplace=True)

    # Wages in the public sector are paid every two weeks on the 15th and on the last day of the month.
    # Supermarket sales could be affected by this.
    d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

    st.write(d)

    # 지진이 매출에 영향을 주었는지
    st.write(d[(d.month.isin([4, 5]))].groupby(["year"]).sales.mean())

    # March
    st.write(pd.pivot_table(d[(d.month.isin([3]))], index="year", columns="family", values="sales", aggfunc="mean"))

    # April - May
    st.write(pd.pivot_table(d[(d.month.isin([4, 5]))], index="year", columns="family", values="sales", aggfunc="mean"))

    # June
    st.write(pd.pivot_table(d[(d.month.isin([6]))], index="year", columns="family", values="sales", aggfunc="mean"))

    # ACF & PACF
    a = d[(d.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")

    ## family 에 대한 sales  정보를 ACF 와 PACF 상관관계
    fig_acf_pacf(a)

    a = d[d.year.isin([2016, 2017])].groupby(["year", "day_of_year"]).sales.mean().reset_index()

    ## 연도에 따라 판매 평균을 비교
    fig_average_sales(a)

    # Simple Moving Average (단순 이동 평균)
    a = train.sort_values(["store_nbr", "family", "date"])
    for i in [20, 30, 45, 60, 90, 120, 365, 730]:
        a["SMA" + str(i) + "_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(16).values
        a["SMA" + str(i) + "_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(30).values
        a["SMA" + str(i) + "_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(60).values

    st.write(a[["sales"] + a.columns[a.columns.str.startswith("SMA")].tolist()].corr())

    b = a[(a.store_nbr == 1)].set_index("date")

    ## 단순이동평균을 보여주는 그래프
    fig_SMA_graph(b)

    # Exponential Moving Average(지수 평균 이동)
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [16, 30, 60, 90]

    a = ewm_features(a, alphas, lags)

    ## 지수평균이동을 보여주는 그래프
    fig_EMA_graph(a)




