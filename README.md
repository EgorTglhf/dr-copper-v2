# Dr. Copper — A Quantitative Analysis Pipeline for Copper

A quantitative research pipeline built around CPER, a copper-futures fund, covering data preparation, factor and regime exploration, time-series analysis, GARCH volatility modelling, XGBoost direction forecasting and systematic backtesting — implemented in Python.

Can market, macroeconomic and volatility inputs improve five-session CPER direction forecasts and support useful long/cash trading?

Five notebooks follow the research from data preparation to final evaluation. Results concern CPER shares, not spot copper or individual futures contracts. The selected pipeline's development advantage **did not generalize** to the final historical period.

**Data → diagnostics/PCA/KMeans → mean-model analysis → GARCH → XGBoost → backtesting**

## Table of Contents

- [Research Design](#research-design)
- [Key Results](#key-results)
- [Pipeline Phases](#pipeline-phases)
  - [Phase 1 — Data, Diagnostics, PCA and KMeans](#phase-1--data-diagnostics-pca-and-kmeans)
  - [Phase 2 — Mean-Model Analysis](#phase-2--mean-model-analysis)
  - [Phase 3 — GARCH Volatility](#phase-3--garch-volatility)
  - [Phase 4 — XGBoost](#phase-4--xgboost)
  - [Phase 5 — Backtesting and Final Evaluation](#phase-5--backtesting-and-final-evaluation)
- [Limitations](#limitations)
- [Further Work](#further-work)
- [Repository Map](#repository-map)
- [Methodological References](#methodological-references)

## Research Design

| Segment | Period | Use |
|---|---|---|
| Data1 | 2012–2016 | Initial training and diagnostics |
| Data2 | 2017–2022 | 72 expanding monthly folds; model/strategy development |
| Data3 forecast origins | January 2023–October 2026 | 46 expanding monthly folds; frozen-procedure evaluation |

- **Target:** `Close[t+5] / Close[t] − 1`, five rows forward in the prepared CPER table. Positive = up; zero = non-up; unavailable outcomes remain missing.
- **Timing:** forecast overnight after all input bars are complete. Monthly classifier fits use only labels whose outcomes have matured by the fit cutoff. Later scheduled fits can use earlier Data3 observations and matured labels.
- **Frozen choices:** selected features and XGBoost hyperparameters were fixed before the Data3 evaluation. Parameters are fitted again monthly; the recipe is unchanged.
- **Coverage:** 1,510 Data2 forecasts, with five terminal outcomes unavailable; 961 Data3 forecasts, all scored. Five November 2026 closes settle terminal labels only, with no November forecast origins or trades.

## Key Results

### Direction accuracy

| Model | Data2: development | Data3: final evaluation |
|---|---:|---:|
| XGBoost | 54.62% | 49.74% |
| Always-up | 51.69% | 54.84% |
| L2 logistic regression | 51.83% | 51.93% |
| Scored forecast origins | 1,505 | 961 |

Data2's paired block-bootstrap 95% intervals for XGBoost's accuracy advantage include zero. They condition on selected predictions and do not account for repeated model selection.

### Data3 trading performance, after costs

| Rule | Net return | CAGR¹ | Max drawdown | Fills |
|---|---:|---:|---:|---:|
| XGBoost daily | 8.57% | 2.18% | −27.71% | 182 |
| XGBoost confirmation | 36.54% | 8.51% | −29.33% | 80 |
| SMA 50/200 | 37.65% | 8.74% | −24.77% | 8 |
| CPER buy-and-hold | 71.84% | 15.25% | −24.77% | 2 |
| Assumed cash | 7.87% | — | 0.00% | 0 |

¹ QuantStats' 252-period annualization convention. Assumed cash accrues at a constant 2% effective annual rate over calendar days; it is illustrative, not historical Treasury performance.

With **zero commission and slippage**, daily and confirmation return **22.74%** and **44.11%**, versus **72.07%** for buy-and-hold. Costs do not explain the full shortfall. On Data3, XGBoost trails always-up by **5.10 percentage points** in accuracy; both trading rules underperform buy-and-hold, including without transaction costs.

## Pipeline Phases

### Phase 1 — Data, Diagnostics, PCA and KMeans

**[Notebook](notebooks/Phase_1_Data_pipeline_PCA.ipynb)**

| Inputs | Source |
|---|---|
| CPER, GLD, USO, S&P 500, DXY, USD/CNY | Yahoo Finance via yfinance |
| 10-year real yield: DFII10 | FRED/ALFRED |
| USA and China amplitude-adjusted CLI | OECD archived monthly editions |

Two disputed CPER rows are removed before feature construction. Retain **19 features**: market 1/5-row log returns, 21-row copper volatility, macro levels and changes. Data1 plots and ADF diagnostics inform the macro representations.

Training-only scaling/PCA refits monthly: **12 PCs retain approximately 83–85%** of training variance. Two KMeans groups have weak separation, with median silhouette about **0.10**. Components and cluster IDs have no guaranteed permanent economic meaning. Both were explored; neither enters the selected classifier.

### Phase 2 — Mean-Model Analysis

**[Notebook](notebooks/Phase_2_Time_series_analysis.ipynb)**

ADF, ACF/PACF and Ljung–Box diagnostics motivate comparing zero mean, constant mean and AR(1)–AR(20). Zero mean has the lowest Data1 BIC. On Data2, AR(1) reduces daily MSE by only **0.03%** and increases five-row MSE by **0.42%** relative to zero mean.

**Decision:** use zero mean for GARCH: added mean dynamics provide no clear development forecast advantage.

### Phase 3 — GARCH Volatility

**[Notebook](notebooks/Phase_3_GARCH_volatility_modelling.ipynb)**

Compare Gaussian and Student-t GARCH(1,1). Student-t improves Data1 BIC, but Gaussian has lower Data2 variance MSE: **4.2% / 16.2% below** the 21-row rolling benchmark at one/five rows. Squared-return proxies are noisy and five-row outcomes overlap.

**Decision:** zero-mean Gaussian GARCH(1,1), monthly parameter refits and daily variance-state updates. Causal one/five-row volatility features begin in 2013. GARCH volatility features were retained, but their contribution to direction accuracy was not evaluated separately.

### Phase 4 — XGBoost

**[Notebook](notebooks/Phase_4_XGB.ipynb)**

Optuna screens four feature groups with **50 trials each**, then runs **350 additional trials** for the winner: **550 trials total**. Every configuration is scored across all 72 Data2 folds.

**Selected:** market + macro + GARCH, **21 columns**, without PCs/regimes; **100 trees, depth 1, learning rate ≈ 0.155**. Exact settings are saved in [selected_direction_config.json](data/processed/direction/selected_direction_config.json). XGBoost and logistic predict up only above 0.5; logistic uses training-only scaling and selected **C=10**. Benchmarks are always-up and L2 logistic regression.

### Phase 5 — Backtesting and Final Evaluation

**[Notebook](notebooks/Phase_5_Backtesting.ipynb)**

| Rule | Entry | Exit |
|---|---|---|
| XGBoost daily | Up forecast | Non-up forecast |
| XGBoost confirmation | Two consecutive up forecasts | Two consecutive non-up forecasts |
| SMA 50/200 | SMA50 > SMA200 | Otherwise |

The five-row prediction horizon does not impose a five-row holding period. Repeated long signals retain one position; overlapping forecasts do not stack exposure.

| Backtest assumption | Value |
|---|---|
| Engine | pandas/NumPy, long/cash |
| Initial capital | $100,000 |
| Execution | Next prepared Close after the signal |
| Entry budget | 100% of available cash, including costs; whole shares |
| Leverage / rebalancing while long | None |
| Commission | $0.005/share; no minimum |
| Slippage | 5 bp adverse per side |
| Uninvested cash | Assumed 2% effective annual rate, calendar days/365 |
| Boundaries | Start in cash with fresh signal state; final-close liquidation |

Confirmation led the Data2 return comparison (**53.86%**, versus buy-and-hold **38.56%**). Final results above reverse that ranking. Logistic remains a predictive benchmark, not a trading rule.

**Where confirmation failed on Data3**

- **Shock exposure:** July 2025 was its worst month (**−11.67%**). It remained long during CPER's **−19.31%** close-to-close fall on July 30, coinciding with the [tariff announcement excluding refined copper](https://www.spglobal.com/market-intelligence/en/news-insights/research/prices-decline-as-us-50-tariff-excludes-refined-copper).
- **Missed rallies:** it held shares over **62.7%** of evaluated close-to-close intervals, but stayed entirely in cash during March and December 2025. CPER gained **11.35% / 8.30%** in those months; the strategy earned only **0.17% / 0.18%** from cash interest. It captured some other sharp rises, including **+8.41%** on July 8.
- **Interpretation:** confirmation missed important rallies without avoiding the largest fall. Its response is delayed by confirmation and next-close execution. Missing demand or policy information is a possible explanation, not an established cause.

**QuantStats reports:** [Data2 daily](reports/phase5_daily_development.html) · [Data2 confirmation](reports/phase5_confirmation_development.html) · [Data3 daily](reports/phase5_daily_final.html) · [Data3 confirmation](reports/phase5_confirmation_final.html)

## Limitations

- **Selection:** repeated Data2 model/strategy comparisons limit development claims. Overlapping targets create dependent errors; one historical path is limited evidence.
- **Source history:** provider history can change; pristine downloads and underlying release snapshots are not persisted. Yahoo Close adjustment/action conventions and historical bar availability are not independently audited execution data. CPER share returns incorporate fund exposure and expenses.
- **Macro availability:** use first nonmissing archived values, a seven-calendar-day yield lag and OECD editions from the following month, with causal forward filling. OECD archives may contain within-month revisions and are not certified original-release-day data. Missing snapshots and publication-time uncertainty remain limitations.
- **Execution and risk:** daily closes cannot establish historical spreads, impact or closing-auction fills. Settlement constraints and exact order mechanics are omitted; sale proceeds fund subsequent entries. QuantStats VaR/ES are descriptive one-period 95% normal-model estimates, not validated tail forecasts or loss limits.

This is a research project, not a production trading system.

## Further Work

- **Demand indicators:** evaluate China/USA manufacturing [PMI, especially new orders](https://www.spglobal.com/market-intelligence/en/solutions/products/resources/pmi-faq), alongside the existing CLI inputs, using actual publication dates.
- **Physical-market indicators:** evaluate changes in [LME copper warehouse stocks, deliveries and cancelled warrants](https://www.lme.com/en/market-data/reports-and-data/warehouse-and-stocks-reports), subject to historical access and release timing. Exchange stocks cover only part of copper inventories.
- **State estimation:** explore Kalman-filter estimates of copper trend and deviations from that trend as additional forecasting features, using only past and current observations.
- **Options information:** evaluate copper-options implied volatility, skew and term structure as features; consider Heston/SABR calibration only with suitable historical option quotes. Futures-option signals require checking their relevance to CPER.
- **Risk-based sizing:** explore exposure caps based on forecast VaR/expected shortfall, supported by Monte Carlo and tariff-like stress scenarios. Test whether reduced downside compensates for missed upside; simulations do not create predictive information or guarantee protection.

These are untested extensions. Assess them on new evaluation data; improvements on the already examined Data3 would remain further development.

## Repository Map

| Location | Contents |
|---|---|
| [Source helpers](src/dr_copper_v2) | Data, features, diagnostics, models and backtest helpers |
| [data/processed](data/processed) | Prepared development/final inputs and terminal outcome closes |
| [PCA artifacts](data/processed/pca/n_components_12) | 72 fitted scaler/PCA/KMeans pipelines, scores, assignments and splits |
| [GARCH artifacts](data/processed/garch) | Causal development volatility forecasts |
| [Direction artifacts](data/processed/direction) | Selected configuration and development/final XGBoost predictions |
| [reports](reports) | Four QuantStats HTML reports |

## Methodological References

- **Ruey S. Tsay, _Analysis of Financial Time Series_, 3rd ed. (2010):** return/mean diagnostics, GARCH and PCA; Chapters 2–3 and §9.4.
- **Marcos López de Prado, _Advances in Financial Machine Learning_ (2018):** chronological evaluation, purging overlapping training labels and walk-forward selection limitations; §§12.2–12.2.1, pp. 161–162.
- **John C. Hull, _Options, Futures, and Other Derivatives_, 11th Global ed. (2022):** VaR and expected-shortfall interpretation; §22.1, pp. 514–516.

These sources support methodological principles. Feature definitions, PCA/KMeans settings, XGBoost search, split dates, trading rules and execution/cash assumptions are project choices.

---

*Built as a quantitative finance portfolio project. Predictive evaluation uses chronological walk-forward folds, with Data2 used for development and Data3 for final evaluation. Simulated past performance does not guarantee future results.*
