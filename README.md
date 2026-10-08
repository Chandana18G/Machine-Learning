# COVID-19 ICU & Mortality Forecasting

M.Sc. coursework project (Applied Data Science & AI, SRH University). Forecasting daily
COVID-19 deaths and ICU occupancy for the **Netherlands**, **Germany**, the German state of
**Bremen** and the **UK**, comparing Linear Regression, Random Forest and XGBoost with
time-series backtesting.

Write-up with charts: [chandana18g.github.io/projects/icu-mortality-forecasting](https://chandana18g.github.io/projects/icu-mortality-forecasting/)

## Approach

Every notebook follows the same recipe:

1. **Backtest first.** Train on data up to 1 December 2020 and test on December 2020 to
   January 2021, to check that the feature set and models behave sensibly.
2. **Forecast.** Train on data up to 1 December 2021 and hold out the last 62 days
   (1 December 2021 to 31 January 2022).
3. **Leakage-safe features.** Only lagged or rolling values the model could have known at
   forecast time: new cases lagged 14 days, a 7-day rolling mean of cases, lagged hospital
   or ICU occupancy, vaccination rate, stringency index and reproduction rate (where the
   source provides them). Targets are 7-day smoothed deaths or ICU patients (Bremen uses raw
   daily deaths).
4. **Three models.** Linear Regression as the baseline, then Random Forest and XGBoost.
   Where cross-validation is used it is `TimeSeriesSplit`, so no fold trains on the future.
5. **Export.** The best model's January 2022 forecast is written to a CSV next to the
   notebook.

## Results

Mean absolute error on the 62-day holdout (1 Dec 2021 to 31 Jan 2022). Lower is better.

| Target | Linear | Random Forest | XGBoost |
| --- | ---: | ---: | ---: |
| Germany, daily deaths (7-day avg) | 345.0 | 52.7 | **46.4** |
| Netherlands, daily deaths (7-day avg) | 10.4 | 9.9 | **6.3** |
| Netherlands, ICU patients (7-day avg) | 91.1 | 39.3 | **34.6** |
| Bremen, daily deaths | 3.0 | 0.44 | **0.43** |
| UK, daily deaths (7-day avg) | n/a | **19.2** | 48.8 |

- In Germany, XGBoost cut the error by **87%** compared with the linear baseline.
- **No single model wins everywhere.** In the UK, Random Forest beat XGBoost, and the
  fold-by-fold cross-validation errors for XGBoost varied widely (standard deviation 61 on a
  mean of 72), so a good average can hide an unstable model.
- The UK notebook keeps only the Random Forest and XGBoost comparison in its final run; the
  linear baseline was run in the backtest (MAE 111.8) but not in the forecast step.

Backtest results (December 2020 to January 2021) are printed inside each notebook.

## Notebooks

| Notebook | Region and target | Data source |
| --- | --- | --- |
| `deathsgermanynew.ipynb` | Germany, daily deaths | Our World in Data |
| `deathsnetherlands.ipynb` | Netherlands, daily deaths | Our World in Data |
| `icu-netherlands.ipynb` | Netherlands, ICU patients | Our World in Data |
| `baselineuktarget1.ipynb` | United Kingdom, daily deaths | Our World in Data |
| `bremen.ipynb` | Bremen (German state), daily deaths | Robert Koch Institute |

Each notebook is self-contained: a backtest section followed by a forecast section.

## Data

The data files are not committed. The notebooks load them from these paths, so either place
the files there or change the path at the top of each notebook:

- `/Users/chandana/Documents/Python for Data Science/compact.csv`: the Our World in Data
  COVID-19 dataset (compact version), available from
  https://docs.owid.io/projects/covid/en/latest/dataset.html. The notebooks use the
  columns `country`, `date`, `new_cases`, `new_deaths`, `icu_patients`, `hosp_patients`,
  `people_fully_vaccinated_per_hundred`, `stringency_index` and `reproduction_rate`.
- `/Users/chandana/Downloads/rki_merged_all_ages.csv`: a merge of the Robert Koch Institute's public daily case,
  death and hospitalisation series by federal state and age group, with the columns
  `Meldedatum`, `Bundesland`, `Altersgruppe`, `Faelle_neu`, `Todesfaelle`,
  `7T_Hospitalisierung_Faelle`, `Inzidenz_7-Tage` and `7T_Hospitalisierung_Inzidenz`. The
  RKI publishes the source tables at https://github.com/robert-koch-institut.

## Setup

```bash
git clone https://github.com/Chandana18G/Machine-Learning.git
cd Machine-Learning

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter.

## What I learned

- Honest time-series validation (backtest, then walk-forward cross-validation) matters more
  than model choice. A random train/test split would have leaked the future into training.
- Look at the error for each fold, not just the mean.
- Tree models handled the regime change of the Omicron wave far better than the linear
  baseline, but which tree model wins depends on the region.

## Author

**Chandana Gurusiddappa**
GitHub: [@Chandana18G](https://github.com/Chandana18G) ·
Portfolio: [chandana18g.github.io](https://chandana18g.github.io)
