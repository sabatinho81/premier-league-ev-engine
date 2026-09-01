# 🦁 Premier League Hybrid Matchday Engine & EV Tracker

<div align="center">

A production-ready quantitative football prediction engine combining venue-specific tactical metrics with real-time market-implied probabilities.

[**Explore Live App**](https://premier-league-hybrid-engine.streamlit.app/)

</div>

---

## 🧠 Architecture & Methodology

Traditional prediction models either rely entirely on raw box-score stats (ignoring crucial market intelligence) or blindly follow bookmaker lines. This engine implements a **Hybrid Architecture** designed to exploit pricing inefficiencies:

1. **Venue-Specific Rolling Statistics:** Evaluates how clubs perform relative to their environment, isolating home and away behaviors rather than relying on blunt aggregate metrics.
2. **Market-Implied Probabilities:** Integrates historical and live bookmaker odds directly into the feature space to anchor onto collective market wisdom before applying tactical adjustments.
3. **Dual-Model Agility:** Powered by **100 sequential decision trees** via `xgboost`, utilizing class balancing (`compute_sample_weight`) to prevent bias across uneven outcome distributions.

---

## ⚡ Key Capabilities

* **Hybrid & Pure Modes:** Toggle bookmaker odds on the fly to run the engine as a pure tactical model or a market-blended hybrid system.
* **Smart Promoted-Team Pipeline:** Automatically handles newly promoted clubs (e.g., Hull, Coventry) without crashing or relying on stale historical defaults by calculating rolling data directly from current-season raw inputs.
* **Real-Time Expected Value (EV):** Instantly computes expected value per wager based on true model probabilities versus live bookmaker quotes, featuring custom risk-management margin sliders.
* **Bilingual Localization:** Full native support for **English** and **Polski**, switchable instantly from the sidebar.
* **Dynamic Currency Selector:** Seamlessly calculate stakes and returns in **GBP (£)**, **EUR (€)**, or **PLN (zł)**.
* **Isolated Performance Verifiers:** Split audit ledgers tracking **Pure Directional Accuracy** independently from **Value Betting ROI**, secured behind a master PIN ledger system.

---

## 📊 Core Feature Matrix

| Feature | Description |
| :--- | :--- |
| **Venue Momentum** | Difference between short-term points (last 3 venue games) and long-term average (last 10 venue games). |
| **Net Goal Diff** | Average goals scored minus goals conceded over the last 10 games at this specific venue. |
| **2nd Half Threat** | Net goals in the second half over the last 10 venue games, capturing fitness and tactical adjustments. |
| **Conversion Rate** | Percentage of shots on target resulting in a goal (Goals Scored / (Shots on Target + 0.01)). |
| **Overall Form (Last 5)** | Cumulative points and goal differences across all venues over the immediate past 5 matches. |

---

## 🛠️ Tech Stack

* **Core Engine:** Python, Pandas, NumPy, Scikit-Learn, XGBoost
* **Web Interface:** Streamlit Cloud (Demo)
* **Data Source:** Football-data.co.uk automated pipelines
* **Persistence:** Local CSV audit ledgers with interactive `st.data_editor` controls

---

## 🚀 Quickstart & Installation

1. **Clone the repository:

   ```bash
   it clone [https://github.com/your-username/pl-hybrid-engine.git](https://github.com/your-username/pl-hybrid-engine.git)

   cd pl-hybrid-engine
   ```
3. **Install dependencies:

    ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit application locally:

   ```bash
   streamlit run app.py
   ```
