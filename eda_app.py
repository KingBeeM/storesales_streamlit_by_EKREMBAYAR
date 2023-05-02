# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import utils
import data_app

def eda_app():
    st.subheader("Exploratory Data Structures")
    train, test, transactions, stores, oil, holidays = data_app.load_data()
    selected_data = st.sidebar.selectbox("SELECT DATA",["Train", "Test", "Transactions", "Stores", "Oil", "Holidays_Events"])

    # Datetime
    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)

    # Data types
    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    # Resample
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

    # Interpolate
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()

    if selected_data == "Train":
        pass
    if selected_data == "Test":
        pass
    if selected_data == "Stores":
        pass
    if selected_data == "Holidays_Events":
        pass

    if selected_data == "Transactions":
        selected_charts = st.selectbox("SELECT CHART", ["Transactions Line Graph", "Transactions Box Graph", "Monthly Average Transactions Graph", "Transactions X Sales Correnlation Graph", "Day of Week Transactions Line Graph"])

        if selected_charts == "Transactions Line Graph":
            temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
            st.write("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
            fig1, ax = plt.subplots()
            fig1 = px.line(transactions.sort_values(["store_nbr", "date"]), x="date", y="transactions", color="store_nbr", title="Transactions")
            st.plotly_chart(fig1)

        if selected_charts == "Transactions Box Graph":
            a = transactions.copy()
            a["year"] = a.date.dt.year
            a["month"] = a.date.dt.month
            fig2, ax = plt.subplots()
            fig2 = px.box(a, x="year", y="transactions", color="month", title="Transactions")
            st.plotly_chart(fig2)

        if selected_charts == "Monthly Average Transactions Graph":
            a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
            a["year"] = a.date.dt.year
            fig3, ax = plt.subplots()
            fig3 = px.line(a, x="date", y="transactions", color="year", title="Monthly Average Transactions")
            st.plotly_chart(fig3)

        if selected_charts == "Transactions X Sales Correnlation Graph":
            fig4, ax = plt.subplots()
            temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
            fig4 = px.scatter(temp, x="transactions", y="sales", trendline="ols", trendline_color_override="red")
            st.plotly_chart(fig4)

        if selected_charts == "Day of Week Transactions Line Graph":
            a = transactions.copy()
            a["year"] = a.date.dt.year
            a["dayofweek"] = a.date.dt.dayofweek + 1
            a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
            fig5, ax = plt.subplots()
            fig5 = px.line(a, x="dayofweek", y="transactions", color="year", title="Transactions")
            st.plotly_chart(fig5)

    if selected_data == "Oil":
        selected_charts = st.selectbox("SELECT CHART", ["Daily Oil Price", "Correnlation with Datily Oil Prices", "Daily Oil Product & Total Family Sales"])

        if selected_charts == "Daily Oil Price":
            p = oil.melt(id_vars=["date"] + list(oil.keys()[5:]), var_name="Legend")
            fig1, ax = plt.subplots()
            fig1 = px.line(p.sort_values(["Legend", "date"], ascending=[False, True]), x="date", y="value", color="Legend", title="Daily Oil Price")
            st.plotly_chart(fig1)

        if selected_charts == "Correnlation with Datily Oil Prices":
            temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
            temp = pd.merge(temp, oil, how="left")
            st.write("Correnlation with Datily Oil Prices")
            st.write(temp.drop(["store_nbr", "dcoilwtico"], axis=1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")

            fig2, ax = plt.subplots(1, 2, figsize=(15, 5))
            temp.plot.scatter(x="dcoilwtico_interpolated", y="transactions", ax=ax[0])
            temp.plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[1], color="r")
            ax[0].set_title("Daily Oil Price & Transactions", fontsize=15)
            ax[1].set_title("Daily Oil Price & Sales", fontsize=15)
            st.pyplot(fig2)

        if selected_charts == "Daily Oil Product & Total Family Sales":
            a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis=1), how="left")
            c = a.groupby("family").corr("spearman").reset_index()
            c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

            fig3, ax = plt.subplots(7, 5, figsize=(20, 20))
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
            st.pyplot(fig3)

    if selected_data == "Train":
        selected_charts = st.selectbox("SELECT CHART", ["Correlation among stores", "Daily Total Sales of The Stores", "zero_prediction", "Daily Total Sales of The Family", "Which Product Family Preferred more?", "City and Stores"])

        if selected_charts == "Correlation among stores":
            a = train[["store_nbr", "sales"]]
            a["ind"] = 1
            a["ind"] = a.groupby("store_nbr").ind.cumsum().values
            a = pd.pivot(a, index="ind", columns="store_nbr", values="sales").corr()

            mask = np.triu(a.corr())
            fig1, ax = plt.subplots(1, 1, figsize=(20, 20))
            sns.heatmap(a,
                        annot=True,
                        fmt=".1f",
                        cmap="coolwarm",
                        square=True,
                        mask=mask,
                        linewidths=1,
                        cbar=False,
                        ax=ax)
            plt.title("Correlation among stores", fontsize=20)
            st.pyplot(fig1)

        if selected_charts == "Daily Total Sales of The Stores":
            a = train.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()
            fig2, ax = plt.subplots()
            fig2 = px.line(a, x="date", y="sales", color="store_nbr", title="Daily Total Sales of The Stores")
            st.plotly_chart(fig2)

        if selected_charts == "zero_prediction":
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

            zero_prediction = []
            for i in range(0, len(c)):
                zero_prediction.append(pd.DataFrame({
                    "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
                    "store_nbr": c.store_nbr.iloc[i],
                    "family": c.family.iloc[i],
                    "sales": 0
                }))
            zero_prediction = pd.concat(zero_prediction)

            c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()

            fig3, ax = plt.subplots(1, 5, figsize=(20, 4))
            train[(train.store_nbr == 10) & (train.family == "LAWN AND GARDEN")].set_index("date").sales.plot(ax=ax[0], title="STORE 10 - LAWN AND GARDEN")
            train[(train.store_nbr == 36) & (train.family == "LADIESWEAR")].set_index("date").sales.plot(ax=ax[1], title="STORE 36 - LADIESWEAR")
            train[(train.store_nbr == 6) & (train.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax=ax[2], title="STORE 6 - SCHOOL AND OFFICE SUPPLIES")
            train[(train.store_nbr == 14) & (train.family == "BABY CARE")].set_index("date").sales.plot(ax=ax[3], title="STORE 14 - BABY CARE")
            train[(train.store_nbr == 53) & (train.family == "BOOKS")].set_index("date").sales.plot(ax=ax[4], title="STORE 43 - BOOKS")
            st.pyplot(fig3)

        if selected_charts == "Daily Total Sales of The Family":
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

            zero_prediction = []
            for i in range(0, len(c)):
                zero_prediction.append(pd.DataFrame({
                    "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
                    "store_nbr": c.store_nbr.iloc[i],
                    "family": c.family.iloc[i],
                    "sales": 0
                }))
            zero_prediction = pd.concat(zero_prediction)

            c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()

            a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()
            fig4, ax = plt.subplots()
            fig4 = px.line(a, x="date", y="sales", color="family", title="Daily Total Sales of The Family")
            st.plotly_chart(fig4)

        if selected_charts == "Which Product Family Preferred more?":
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

            zero_prediction = []
            for i in range(0, len(c)):
                zero_prediction.append(pd.DataFrame({
                    "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
                    "store_nbr": c.store_nbr.iloc[i],
                    "family": c.family.iloc[i],
                    "sales": 0
                }))
            zero_prediction = pd.concat(zero_prediction)

            c = train.groupby(["family", "store_nbr"]).tail(60).groupby(
                ["family", "store_nbr"]).sales.sum().reset_index()

            a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()
            a = train.groupby("family").sales.mean().sort_values(ascending=False).reset_index()
            fig5, ax = plt.subplots()
            fig5 = px.bar(a, y="family", x="sales", color="family", title="Which Product Family Preferred more?")
            st.plotly_chart(fig5)

        if selected_charts == "City and Stores":
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

            d = pd.merge(train, stores)
            d["store_nbr"] = d["store_nbr"].astype("int8")
            d["year"] = d.date.dt.year

            fig6, ax = plt.subplots()
            fig6 = px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x="year", y="sales", color="city")
            st.plotly_chart(fig6)
