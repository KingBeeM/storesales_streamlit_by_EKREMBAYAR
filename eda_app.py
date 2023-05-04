# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import shapiro
import scipy.stats as stats
import utils


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

def AB_Test(dataframe, group, target):
    # Split A/B
    groupA = dataframe[dataframe[group]==1][target]
    groupB = dataframe[dataframe[group]==0][target]

    if len(groupA) < 3:
        print("Not enough data.")
        return None

    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0 : Distribution is Normal - False
    # H1 : Distaribution is not Normal - True

    if (ntA == False) & (ntB == False): # H0 : Normal Distribution
        # Parametic Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0 : Homogeneity - False
        # H1 : Heterogeneous: True

        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0 : M1 == M2 - False
            # H1 : M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0 : M1 == M2 - False
            # H1 : M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1]
        # H0 : M1 == M2 - False
        # H1 : M1 != M2 - True

    #Result
    temp = pd.DataFrame({
        "AB Hypothesis" : [ttest < 0.05],
        "p-value" : [ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar", "A/B groups are not similar")
    temp["Feature"] = group
    temp["GroupA_mean"] = groupA.mean()
    temp["GroupB_mean"] = groupB.mean()
    temp["GroupA_median"] = groupA.median()
    temp["GroupB_median"] = groupB.median()

    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Feature", "Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
    else:
        temp = temp[["Feature", "Test Type", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]

    return temp

def fig_Transactions_TotalSales_Correlation(temp, transactions):
    """
    Transactions 데이터와 Total Sales 간의 상관관계 패턴 파악 하는 그래프
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
    st.write("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))

    fig, ax = plt.subplots()
    fig = px.line(transactions.sort_values(["store_nbr", "date"]), x="date", y="transactions", color="store_nbr", title="Transactions")
    st.plotly_chart(fig)

def fig_Transactions_ym_patten1(transactions):
    """
    Transactions 데이터의 연도별, 월별 패턴 파악 하는 그래프
    """
    a = transactions.copy()
    a["year"] = a.date.dt.year
    a["month"] = a.date.dt.month

    fig, ax = plt.subplots()
    fig = px.box(a, x="year", y="transactions", color="month", title="Transactions")
    st.plotly_chart(fig)

def fig_Transactions_ym_patten2(transactions):
    """
    Transactions 데이터의 연도별, 월별 평균 매출 패턴 파악 하는 그래프
    """
    a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
    a["year"] = a.date.dt.year

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="transactions", color="year", title="Monthly Average Transactions")
    st.plotly_chart(fig)

def fig_Transactions_Sales_Correlation(temp):
    """
    Transactions 데이터와 Sales 간의 상관관계 패턴 파악 하는 그래프
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")

    fig, ax = plt.subplots()
    fig = px.scatter(temp, x="transactions", y="sales", trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig)

def fig_Transactions_ydw_patten(transactions):
    """
    Transactions 연도별, 요일별 패턴 파악 하는 그래프
    """
    a = transactions.copy()
    a["year"] = a.date.dt.year
    a["dayofweek"] = a.date.dt.dayofweek + 1
    a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="dayofweek", y="transactions", color="year", title="Transactions")
    st.plotly_chart(fig)

def fig_OilPrice(oil):
    """
    Oil Price 누락 값 추가 하는 그래프
    """
    # oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    # oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    # oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
    p = oil.melt(id_vars=["date"] + list(oil.keys()[5:]), var_name="Legend")

    fig, ax = plt.subplots()
    fig = px.line(p.sort_values(["Legend", "date"], ascending=[False, True]), x="date", y="value", color="Legend", title="Daily Oil Price")
    st.plotly_chart(fig)

def fig_OilPrice_Sales_Transactions_patten(temp, oil):
    """
    Oil Price 와 Sales / Oil Price 와 Transactions 패턴 파악 하는 그래프
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
    temp = pd.merge(temp, oil, how="left")
    st.write("Correnlation with Daily Oil Prices")
    st.write(temp.drop(["store_nbr", "dcoilwtico"], axis=1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    temp.plot.scatter(x="dcoilwtico_interpolated", y="transactions", ax=ax[0])
    temp.plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[1], color="r")
    ax[0].set_title("Daily Oil Price & Transactions", fontsize=15)
    ax[1].set_title("Daily Oil Price & Sales", fontsize=15)
    st.pyplot(fig)

def fig_OilPrice_family_patten(train, oil):
    """
    Oil Price 와 제품군 별 Sales 패턴 파악 하는 그래프
    """
    a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis=1), how="left")
    c = a.groupby("family").corr("spearman").reset_index()
    c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

    fig, ax = plt.subplots(7, 5, figsize=(20, 20))
    for i, fam in enumerate(c.family):
        if i < 6:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[0, i - 1])
            ax[0, i - 1].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[0, i - 1].axvline(x=70, color="r", linestyle="--")
        if i >= 6 and i < 11:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[1, i - 6])
            ax[1, i - 6].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[1, i - 6].axvline(x=70, color='r', linestyle='--')
        if i >= 11 and i < 16:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[2, i - 11])
            ax[2, i - 11].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[2, i - 11].axvline(x=70, color='r', linestyle='--')
        if i >= 16 and i < 21:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[3, i - 16])
            ax[3, i - 16].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[3, i - 16].axvline(x=70, color='r', linestyle='--')
        if i >= 21 and i < 26:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[4, i - 21])
            ax[4, i - 21].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[4, i - 21].axvline(x=70, color='r', linestyle='--')
        if i >= 26 and i < 31:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[5, i - 26])
            ax[5, i - 26].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[5, i - 26].axvline(x=70, color='r', linestyle='--')
        if i >= 31:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[6, i - 31])
            ax[6, i - 31].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[6, i - 31].axvline(x=70, color='r', linestyle='--')

    plt.tight_layout(pad=5)
    plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize=20)
    st.pyplot(fig)

def fig_Train_sales_Correlation(train):
    """
    각 매장별 Sales 에 대한 상관 관계 그래프
    """
    a = train[["store_nbr", "sales"]]
    a["ind"] = 1
    a["ind"] = a.groupby("store_nbr").ind.cumsum().values
    a = pd.pivot(a, index="ind", columns="store_nbr", values="sales").corr()

    mask = np.triu(a.corr())
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    sns.heatmap(a, annot=True, fmt=".1f", cmap="coolwarm", square=True, mask=mask, linewidths=1, cbar=False)
    plt.title("Correlation among stores", fontsize=20)
    st.pyplot(fig)

def fig_Train_store_TotalSales_patten(train):
    """
    각 매장 별 Total Sales 패턴 파악
    """
    a = train.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="sales", color="store_nbr", title="Daily Total Sales of The Stores")
    st.plotly_chart(fig)


def fig_unsold_family(train):
    """
    판매 되지 않는 제품 군 파악 하는 그래프
    """
    c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    train[(train.store_nbr == 10) & (train.family == "LAWN AND GARDEN")].set_index("date").sales.plot(ax=ax[0], title="STORE 10 - LAWN AND GARDEN")
    train[(train.store_nbr == 36) & (train.family == "LADIESWEAR")].set_index("date").sales.plot(ax=ax[1], title="STORE 36 - LADIESWEAR")
    train[(train.store_nbr == 6) & (train.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax=ax[2], title="STORE 6 - SCHOOL AND OFFICE SUPPLIES")
    train[(train.store_nbr == 14) & (train.family == "BABY CARE")].set_index("date").sales.plot(ax=ax[3], title="STORE 14 - BABY CARE")
    train[(train.store_nbr == 53) & (train.family == "BOOKS")].set_index("date").sales.plot(ax=ax[4], title="STORE 43 - BOOKS")
    st.pyplot(fig)

def fig_Train_d_family_patten(train):
    """
    일별 제품 판매 패턴 파악 그래프
    """
    a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="sales", color="family", title="Daily Total Sales of The Family")
    st.plotly_chart(fig)

def fig_Train_family_patten(train):
    """
    제품별 판매 패턴 파악 그래프
    """
    a = train.groupby("family").sales.mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots()
    fig = px.bar(a, y="family", x="sales", color="family", title="Which Product Family Preferred more?")
    st.plotly_chart(fig)

def fig_Train_Stores_patten(train, stores):
    """
    매장 별 판매 패턴 파악 그래프
    """
    d = pd.merge(train, stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")
    d["year"] = d.date.dt.year

    fig, ax = plt.subplots()
    fig = px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x="year", y="sales", color="city")
    st.plotly_chart(fig)

def Feature_Engineering_Holidays(holidays, train, test, stores):
    """
    휴일 데이터에 대해서 전처리 하는 부분
    """
    ## Transferred Holidays(양도된 휴일) 처리
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1).reset_index(drop=True)
    tr = pd.concat([tr1, tr2], axis=1)
    tr = tr.iloc[:, [5, 1, 2, 3, 4]]

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
    holidays = pd.concat([holidays, tr]).reset_index(drop=True)

    ## Additional Holidays(추가된 휴일) 처리
    holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace("\d+","")
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    ## Bridge Holidays(브릿지 휴일) 처리
    holidays["description"] = holidays["description"].str.replace("Puente ", "")
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

    ## Work Day Holidays(근무 휴일(보상 휴일)) 처리
    work_day = holidays[holidays.type == "Work Day"]
    holidays = holidays[holidays.type != "Work Day"]

    ## Events are national(전국 행사) 처리
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis=1).rename({"description": "events"}, axis=1)

    holidays = holidays[holidays.type != "Event"].drop("type", axis=1)
    regional = holidays[holidays.locale == "Regional"].rename({"locale_name": "state", "description": "holiday_regional"}, axis=1).drop("locale", axis=1).drop_duplicates()
    national = holidays[holidays.locale == "National"].rename({"description": "holiday_national"}, axis=1).drop(["locale", "locale_name"], axis=1).drop_duplicates()
    local = holidays[holidays.locale == "Local"].rename({"description": "holiday_local", "locale_name": "city"}, axis=1).drop("locale", axis=1).drop_duplicates()

    d = pd.merge(pd.concat([train, test]), stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")
    ## National Holidays & Events(공휴일 및 이벤트)
    d = pd.merge(d, national, how="left")
    ## Regional(state 별)
    d = pd.merge(d, regional, how="left", on=["date", "state"])
    ## Local(city 별)
    d = pd.merge(d, local, how="left", on=["date", "city"])
    ## Work Day(실제 근무일 컬럼이 생성되면 제거)
    d = pd.merge(d, work_day[["date", "type"]].rename({"type": "IsWorkDay"}, axis=1), how="left")
    ## EVENT
    events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how="left")
    d[events_cat] = d[events_cat].fillna(0)

    ## NEW features
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

    d["national_independence"] = np.where(d.holiday_national.isin(["Batalla de Pichincha", "Independencia de Cuenca", "Independencia de Guayaquil", "Independecia de Guayaquil", "Primer Grito de Independencia"]), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

    holidays, holidays_cat = one_hot_encoder(d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis=1), holidays], axis=1)

    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + \
              d.columns[d.columns.str.startswith("local")].tolist()
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    return d

def eda_app():
    train, test, transactions, stores, oil, holidays = utils.load_data()
    selected_data = st.sidebar.selectbox("SELECT DATA",["Train", "Transactions", "Oil", "Holidays_Events"])
    st.subheader(f"Exploratory Data Structures - {selected_data} DATA")

    # Datetime
    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)
    holidays["date"] = pd.to_datetime(holidays.date)

    # Data types
    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    # Transactions
    temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")

    if selected_data == "Transactions":
        selected_chart = st.sidebar.selectbox("SELECT Chart",["1", "2", "3", "4", "5"])

        if selected_chart == "1":
            ## Transactions 과 Total Sales 간의 상관관계 패턴 파악
            fig_Transactions_TotalSales_Correlation(temp, transactions)
        if selected_chart == "2":
            ## Transactions 연도별, 월별 패턴 파악
            fig_Transactions_ym_patten1(transactions)
        if selected_chart == "3":
            ## Transactions 연도별, 월 평균 매출 패턴 파악
            fig_Transactions_ym_patten2(transactions)
        if selected_chart == "4":
            ## Transactions 와 Sales 간의 상관관계 그래프
            fig_Transactions_Sales_Correlation(temp)
        if selected_chart == "5":
            ## Transactions 연도별, 요일별 패턴 파악
            fig_Transactions_ydw_patten(transactions)

    # Oil
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()

    if selected_data == "Oil":
        selected_chart = st.sidebar.selectbox("SELECT Chart", ["1", "2", "3"])

        if selected_chart == "1":
            ## Oil Price 누락 값 추가
            fig_OilPrice(oil)
        if selected_chart == "2":
            ## Oil Price 와 Sales / Transactions 패턴 파악
            fig_OilPrice_Sales_Transactions_patten(temp, oil)
        if selected_chart == "3":
            ## Oil Price 와 제품군 별 Sales 패턴 파악
            fig_OilPrice_family_patten(train, oil)


    # Sales
    if selected_data == "Train":
        selected_chart = st.sidebar.selectbox("SELECT Chart", ["1", "2", "3", "4", "5"])

#        if selected_chart == "1":
#            # 각 매장별 Sales 에 대한 상관 관계 그래프
#            fig_Train_sales_Correlation(train)
        if selected_chart == "1":
            # 각 매장 별 Total Sales 패턴 파악
            fig_Train_store_TotalSales_patten(train)

        ## 이상치 제거 : 매장별로 오픈하기 전의 시점
        train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
        train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
        train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
        train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
        train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
        train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
        train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
        train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]

        ## 불필요한 값 제거 : 매장별로 판매하지 않는 제품 파악
        c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family", "store_nbr"])
        c = c[c.sales == 0]

        outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how="outer", indicator=True)
        train = outer_join[~(outer_join._merge == "both")].drop("_merge", axis=1)

        zero_prediction = []
        for i in range(0, len(c)):
            zero_prediction.append(pd.DataFrame({
                "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
                "store_nbr": c.store_nbr.iloc[i],
                "family": c.family.iloc[i],
                "sales": 0
            }))
        zero_prediction = pd.concat(zero_prediction)

        if selected_chart == "2":
            ## 판매 되지 않는 제품군 파악
            fig_unsold_family(train)

        if selected_chart == "3":
            ## 일별 제품 판매 패턴 파악
            fig_Train_d_family_patten(train)

        if selected_chart == "4":
            ## 제품별 판매 패턴 파악
            fig_Train_family_patten(train)

        if selected_chart == "5":
            ## 매장 별 판매 패턴
            fig_Train_Stores_patten(train, stores)

    # Holidays and Events
    d = Feature_Engineering_Holidays(holidays, train, test, stores)
    if selected_data == "Holidays_Events":
        selected_chart = st.sidebar.selectbox("SELECT Chart", ["1", "2", "3"])

        if selected_chart == "1":
            st.write(d)

        if selected_chart == "2":
            ## Apply A/B Testing
            he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + \
                      d.columns[d.columns.str.startswith("local")].tolist()

            ab = []
            for i in he_cols:
                ab.append(AB_Test(dataframe=d[d.sales.notnull()], group=i, target="sales"))
            ab = pd.concat(ab)
            st.write(ab)

        if selected_chart == "3":
            ## Events(Futbol) 과 제품군 패턴
            st.write(d.groupby(["family", "events_Futbol"]).sales.mean()[:60])
