from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ACCOUNTS_FILE = Path("GTM-Engineer Challenge-Daniel - Accounts.csv")
REPS_FILE = Path("GTM-Engineer Challenge-Daniel - Reps.csv")

REQUIRED_ACCOUNT_COLUMNS = {
    "Account_ID",
    "Account_Name",
    "Current_Rep",
    "ARR",
    "Location",
    "Num_Employees",
    "Num_Marketers",
    "Risk_Score",
}
REQUIRED_REP_COLUMNS = {"Rep_Name", "Location", "Segment"}


@dataclass(frozen=True)
class BalanceSummary:
    average_arr: float
    max_arr: float
    min_arr: float
    spread: float


st.set_page_config(
    page_title="Territory Slicer",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_csv(ACCOUNTS_FILE)
    reps = pd.read_csv(REPS_FILE)

    missing_account_columns = REQUIRED_ACCOUNT_COLUMNS - set(accounts.columns)
    missing_rep_columns = REQUIRED_REP_COLUMNS - set(reps.columns)

    if missing_account_columns:
        raise ValueError(
            f"Accounts file is missing columns: {', '.join(sorted(missing_account_columns))}"
        )

    if missing_rep_columns:
        raise ValueError(
            f"Reps file is missing columns: {', '.join(sorted(missing_rep_columns))}"
        )

    accounts["ARR"] = pd.to_numeric(accounts["ARR"], errors="coerce").fillna(0)
    accounts["Num_Employees"] = (
        pd.to_numeric(accounts["Num_Employees"], errors="coerce").fillna(0).astype(int)
    )
    accounts["Num_Marketers"] = (
        pd.to_numeric(accounts["Num_Marketers"], errors="coerce").fillna(0).astype(int)
    )
    accounts["Risk_Score"] = pd.to_numeric(
        accounts["Risk_Score"], errors="coerce"
    ).fillna(0)
    accounts["Segment"] = ""
    reps["Segment"] = reps["Segment"].str.strip()

    return accounts, reps


def assign_segment(accounts: pd.DataFrame, threshold: int) -> pd.DataFrame:
    segmented = accounts.copy()
    segmented["Segment"] = segmented["Num_Employees"].apply(
        lambda employees: "Enterprise" if employees >= threshold else "Mid Market"
    )
    return segmented


def balance_accounts(accounts: pd.DataFrame, rep_names: list[str]) -> pd.DataFrame:
    if accounts.empty:
        return accounts.assign(New_Rep=pd.Series(dtype="str"), Assigned_Rep_ARR=0)

    if not rep_names:
        return accounts.assign(New_Rep="No eligible rep", Assigned_Rep_ARR=0)

    rep_loads = {rep_name: 0.0 for rep_name in rep_names}
    assignments = []

    sorted_accounts = accounts.sort_values(
        by=["ARR", "Num_Employees", "Account_Name"],
        ascending=[False, False, True],
    )

    for _, account in sorted_accounts.iterrows():
        selected_rep = min(rep_loads, key=rep_loads.get)
        rep_loads[selected_rep] += float(account["ARR"])

        assigned = account.to_dict()
        assigned["New_Rep"] = selected_rep
        assigned["Assigned_Rep_ARR"] = rep_loads[selected_rep]
        assignments.append(assigned)

    return pd.DataFrame(assignments)


def reassign_accounts(accounts: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    enterprise_reps = reps.loc[reps["Segment"] == "Enterprise", "Rep_Name"].tolist()
    mid_market_reps = reps.loc[reps["Segment"] == "Mid Market", "Rep_Name"].tolist()

    enterprise_accounts = accounts[accounts["Segment"] == "Enterprise"]
    mid_market_accounts = accounts[accounts["Segment"] == "Mid Market"]

    return pd.concat(
        [
            balance_accounts(enterprise_accounts, enterprise_reps),
            balance_accounts(mid_market_accounts, mid_market_reps),
        ],
        ignore_index=True,
    )


def summarize_balance(rep_summary: pd.DataFrame) -> BalanceSummary:
    if rep_summary.empty:
        return BalanceSummary(0, 0, 0, 0)

    max_arr = float(rep_summary["ARR"].max())
    min_arr = float(rep_summary["ARR"].min())

    return BalanceSummary(
        average_arr=float(rep_summary["ARR"].mean()),
        max_arr=max_arr,
        min_arr=min_arr,
        spread=max_arr - min_arr,
    )


def build_before_after_comparison(
    assignments: pd.DataFrame, reps: pd.DataFrame
) -> pd.DataFrame:
    rep_frame = reps[["Rep_Name", "Segment"]].rename(
        columns={"Rep_Name": "Rep", "Segment": "Rep Segment"}
    )

    current = (
        assignments.groupby("Current_Rep", as_index=False)
        .agg(
            Current_ARR=("ARR", "sum"),
            Current_Accounts=("Account_ID", "count"),
            Current_Avg_Risk=("Risk_Score", "mean"),
            Current_Marketers=("Num_Marketers", "sum"),
        )
        .rename(columns={"Current_Rep": "Rep"})
    )

    proposed = (
        assignments.groupby("New_Rep", as_index=False)
        .agg(
            New_ARR=("ARR", "sum"),
            New_Accounts=("Account_ID", "count"),
            New_Avg_Risk=("Risk_Score", "mean"),
            New_Marketers=("Num_Marketers", "sum"),
        )
        .rename(columns={"New_Rep": "Rep"})
    )

    moved_out = (
        assignments[assignments["Current_Rep"] != assignments["New_Rep"]]
        .groupby("Current_Rep", as_index=False)
        .agg(
            Accounts_Moved_Out=("Account_ID", "count"),
            ARR_Moved_Out=("ARR", "sum"),
        )
        .rename(columns={"Current_Rep": "Rep"})
    )

    moved_in = (
        assignments[assignments["Current_Rep"] != assignments["New_Rep"]]
        .groupby("New_Rep", as_index=False)
        .agg(
            Accounts_Moved_In=("Account_ID", "count"),
            ARR_Moved_In=("ARR", "sum"),
        )
        .rename(columns={"New_Rep": "Rep"})
    )

    comparison = (
        rep_frame.merge(current, on="Rep", how="left")
        .merge(proposed, on="Rep", how="left")
        .merge(moved_out, on="Rep", how="left")
        .merge(moved_in, on="Rep", how="left")
        .fillna(0)
    )

    comparison["ARR_Change"] = comparison["New_ARR"] - comparison["Current_ARR"]
    comparison["Account_Change"] = (
        comparison["New_Accounts"] - comparison["Current_Accounts"]
    )
    comparison["Avg_Risk_Change"] = (
        comparison["New_Avg_Risk"] - comparison["Current_Avg_Risk"]
    )
    comparison["Marketer_Change"] = (
        comparison["New_Marketers"] - comparison["Current_Marketers"]
    )
    comparison["Accounts_Moved"] = (
        comparison["Accounts_Moved_Out"] + comparison["Accounts_Moved_In"]
    )
    comparison["ARR_Moved"] = (
        comparison["ARR_Moved_Out"] + comparison["ARR_Moved_In"]
    )

    return comparison.sort_values(["Rep Segment", "Rep"]).reset_index(drop=True)


def currency(value: float) -> str:
    return f"${value:,.0f}"


def style_directional_changes(value: float, higher_is_better: bool) -> str:
    if value == 0:
        return ""

    is_good = value > 0 if higher_is_better else value < 0
    color = "#116329" if is_good else "#B42318"
    background = "#E7F6EC" if is_good else "#FDE8E7"
    return f"color: {color}; background-color: {background}; font-weight: 700;"


def main() -> None:
    st.title("Territory Slicer")
    st.caption("Model Enterprise vs. Mid Market definitions and rebalance territories by ARR.")

    accounts, reps = load_data()

    min_employees = min(500, int(accounts["Num_Employees"].min()))
    max_employees = int(accounts["Num_Employees"].max())
    default_threshold = int(accounts["Num_Employees"].median())

    with st.sidebar:
        st.header("Territory Controls")
        threshold = st.slider(
            "Enterprise employee threshold",
            min_value=min_employees,
            max_value=max_employees,
            value=default_threshold,
            step=max(1, round((max_employees - min_employees) / 200)),
        )
        st.write(f"Accounts with **{threshold:,}+ employees** are Enterprise.")

        st.divider()
        st.subheader("Rep Pool")
        st.dataframe(
            reps.sort_values(["Segment", "Rep_Name"]),
            hide_index=True,
            use_container_width=True,
        )

    segmented_accounts = assign_segment(accounts, threshold)
    assignments = reassign_accounts(segmented_accounts, reps)

    rep_summary = (
        assignments.groupby(["Segment", "New_Rep"], as_index=False)
        .agg(ARR=("ARR", "sum"), Accounts=("Account_ID", "count"))
        .sort_values(["Segment", "ARR"], ascending=[True, False])
    )
    comparison = build_before_after_comparison(assignments, reps)
    balance = summarize_balance(rep_summary)

    enterprise_accounts = assignments[assignments["Segment"] == "Enterprise"]
    mid_market_accounts = assignments[assignments["Segment"] == "Mid Market"]
    moved_accounts = assignments[assignments["Current_Rep"] != assignments["New_Rep"]]

    metrics = st.columns(5)
    metrics[0].metric("Enterprise Accounts", f"{len(enterprise_accounts):,}")
    metrics[1].metric("Mid Market Accounts", f"{len(mid_market_accounts):,}")
    metrics[2].metric("Enterprise ARR", currency(enterprise_accounts["ARR"].sum()))
    metrics[3].metric("Mid Market ARR", currency(mid_market_accounts["ARR"].sum()))
    metrics[4].metric("Rep ARR Spread", currency(balance.spread))

    movement_metrics = st.columns(4)
    movement_metrics[0].metric("Accounts Moved", f"{len(moved_accounts):,}")
    movement_metrics[1].metric("ARR Moved", currency(moved_accounts["ARR"].sum()))
    movement_metrics[2].metric(
        "Avg Current Risk", f"{assignments['Risk_Score'].mean():.1f}"
    )
    movement_metrics[3].metric(
        "Total Marketers", f"{assignments['Num_Marketers'].sum():,}"
    )

    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("ARR Load by Rep")
        arr_chart = px.bar(
            rep_summary,
            x="New_Rep",
            y="ARR",
            color="Segment",
            text="ARR",
            color_discrete_map={
                "Enterprise": "#28536B",
                "Mid Market": "#C36F09",
            },
            labels={"New_Rep": "Rep", "ARR": "Assigned ARR"},
        )
        arr_chart.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        arr_chart.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            xaxis_title=None,
            legend_title_text=None,
            margin=dict(t=20, r=20, b=20, l=20),
        )
        st.plotly_chart(arr_chart, use_container_width=True)

    with right:
        st.subheader("Account Count by Rep")
        count_chart = px.bar(
            rep_summary,
            x="Accounts",
            y="New_Rep",
            color="Segment",
            orientation="h",
            text="Accounts",
            color_discrete_map={
                "Enterprise": "#28536B",
                "Mid Market": "#C36F09",
            },
            labels={"New_Rep": "Rep"},
        )
        count_chart.update_layout(
            xaxis_title="Accounts",
            yaxis_title=None,
            legend_title_text=None,
            margin=dict(t=20, r=20, b=20, l=20),
        )
        st.plotly_chart(count_chart, use_container_width=True)

    st.subheader("Before vs. After by Rep")
    st.caption(
        "Current ownership comes from Current_Rep. Proposed ownership comes from the ARR-balanced assignment for the selected threshold. Risk and marketers are context only, not assignment inputs."
    )

    comparison_columns = [
        "Rep",
        "Rep Segment",
        "Current_ARR",
        "New_ARR",
        "ARR_Change",
        "Current_Accounts",
        "New_Accounts",
        "Account_Change",
        "Accounts_Moved",
        "ARR_Moved",
        "Current_Avg_Risk",
        "New_Avg_Risk",
        "Avg_Risk_Change",
        "Current_Marketers",
        "New_Marketers",
        "Marketer_Change",
    ]
    styled_comparison = comparison[comparison_columns].style.map(
        lambda value: style_directional_changes(value, higher_is_better=True),
        subset=["ARR_Change"],
    ).map(
        lambda value: style_directional_changes(value, higher_is_better=False),
        subset=["Avg_Risk_Change"],
    )

    st.dataframe(
        styled_comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Current_ARR": st.column_config.NumberColumn("Current ARR", format="$%d"),
            "New_ARR": st.column_config.NumberColumn("New ARR", format="$%d"),
            "ARR_Change": st.column_config.NumberColumn("ARR Change", format="$%d"),
            "Current_Accounts": st.column_config.NumberColumn("Current Accts"),
            "New_Accounts": st.column_config.NumberColumn("New Accts"),
            "Account_Change": st.column_config.NumberColumn("Acct Change"),
            "Accounts_Moved": st.column_config.NumberColumn("Accts Moved"),
            "ARR_Moved": st.column_config.NumberColumn("ARR Moved", format="$%d"),
            "Current_Avg_Risk": st.column_config.NumberColumn(
                "Current Avg Risk", format="%.1f"
            ),
            "New_Avg_Risk": st.column_config.NumberColumn(
                "New Avg Risk", format="%.1f"
            ),
            "Avg_Risk_Change": st.column_config.NumberColumn(
                "Risk Change", format="%.1f"
            ),
            "Current_Marketers": st.column_config.NumberColumn(
                "Current Marketers", format="%d"
            ),
            "New_Marketers": st.column_config.NumberColumn(
                "New Marketers", format="%d"
            ),
            "Marketer_Change": st.column_config.NumberColumn(
                "Marketer Change", format="%d"
            ),
        },
    )

    change_chart = px.bar(
        comparison,
        x="Rep",
        y="ARR_Change",
        color="Rep Segment",
        color_discrete_map={
            "Enterprise": "#28536B",
            "Mid Market": "#C36F09",
        },
        labels={"ARR_Change": "ARR Change", "Rep": "Rep"},
    )
    change_chart.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_title=None,
        legend_title_text=None,
        margin=dict(t=20, r=20, b=20, l=20),
    )
    st.plotly_chart(change_chart, use_container_width=True)

    st.subheader("Threshold Impact")
    scatter = px.scatter(
        assignments,
        x="Num_Employees",
        y="ARR",
        color="Segment",
        hover_name="Account_Name",
        hover_data=["Account_ID", "Current_Rep", "New_Rep", "Location"],
        color_discrete_map={
            "Enterprise": "#28536B",
            "Mid Market": "#C36F09",
        },
        labels={"Num_Employees": "Employees", "ARR": "ARR"},
    )
    scatter.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#333333",
        annotation_text=f"Threshold: {threshold:,}",
        annotation_position="top right",
    )
    scatter.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        margin=dict(t=20, r=20, b=20, l=20),
    )
    st.plotly_chart(scatter, use_container_width=True)

    with st.expander("Assignment logic"):
        st.write(
            """
            Accounts are first segmented using the employee threshold. Within each segment,
            accounts are sorted by ARR from largest to smallest. Each account is then assigned
            to the eligible rep with the lowest current assigned ARR. This balances revenue
            potential directly, while placing the largest accounts early so one rep is less
            likely to receive an outsized territory by accident. Risk score and marketer count
            are shown as context in the before/after view, but they do not influence assignment.
            """
        )


if __name__ == "__main__":
    main()
