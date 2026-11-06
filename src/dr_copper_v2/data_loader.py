import io
import urllib.parse
import urllib.request
from pathlib import Path

import fredapi as fa
import pandas as pd
import yfinance as yf
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]

MARKET_SYMBOLS: dict[str, str] = {
    "copper": "CPER",
    "gold": "GLD",
    "crude": "USO",
    "sp500": "^GSPC",
    "dxy": "DX-Y.NYB",
    "USDCNY": "CNY=X",
}

MACRO_SYMBOLS: dict[str, str] = {
    "real_yield_10y": "DFII10",
}
CLI_COUNTRIES: dict[str, str] = {"cli_usa": "USA", "cli_china": "CHN"}
OECD_API = (
    "https://sdmx.oecd.org/public/rest/v1/data/"
    "OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0/"
)
RELEASE_COLUMNS = [
    "feature",
    "observation_date",
    "vintage_date",
    "available_date",
    "value",
    "source",
]


def _fred_client() -> fa.Fred:
    config = dotenv_values(ROOT / ".env")
    return fa.Fred(api_key=config["FRED_API_KEY"])


def _first_observations(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("observation_date", "vintage_date", "available_date"):
        out[col] = pd.to_datetime(out[col])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    return (
        out.sort_values(["feature", "observation_date", "vintage_date"])
        .drop_duplicates(["feature", "observation_date"], keep="first")
        .loc[:, RELEASE_COLUMNS]
        .reset_index(drop=True)
    )


def _fetch_yf(
    tickers: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    prices = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    return (
        prices["Close"]  # type: ignore
        .rename(columns={symbol: name for name, symbol in tickers.items()})
        .sort_index()
    )


def _fetch_fred(
    tickers: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    fred = _fred_client()
    for feature, symbol in tickers.items():
        cursor = first
        while cursor < stop:
            # FRED limits the number of vintage dates per request. Five years
            # stays below that limit for a daily series; keep chunk seeds until
            # selecting the earliest version across all chunks.
            last = min(cursor + pd.DateOffset(years=5), stop) - pd.Timedelta(days=1)
            lo, hi = cursor.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")
            frame = fred.get_series_all_releases(
                symbol, realtime_start=lo, realtime_end=hi
            )
            for col in ("date", "realtime_start"):
                frame[col] = pd.to_datetime(frame[col])
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.loc[frame.date.ge(first)].copy()
            frames.append(frame.assign(feature=feature, source=f"ALFRED:{symbol}"))
            cursor = last + pd.Timedelta(days=1)
    out = pd.concat(frames, ignore_index=True).rename(
        columns={"date": "observation_date", "realtime_start": "vintage_date"}
    )
    out["available_date"] = out.vintage_date + pd.Timedelta(days=1)
    return _first_observations(out)


def _fetch_cli(
    countries: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    last_edition = (stop - pd.Timedelta(days=1)).to_period("M") - 1
    editions = pd.period_range(first.to_period("M"), last_edition, freq="M")
    key = "+".join(countries.values()) + ".M.LI_AA.IX._T."
    key += "+".join(editions.strftime("%Y%m"))
    url = (
        OECD_API
        + key
        + "?"
        + urllib.parse.urlencode(
            {
                "startPeriod": first.strftime("%Y-%m"),
                "endPeriod": (stop - pd.Timedelta(days=1)).strftime("%Y-%m"),
                "format": "csvfilewithlabels",
            }
        )
    )
    request = urllib.request.Request(url, headers={"User-Agent": "dr-copper-v2/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
    frame = pd.read_csv(io.BytesIO(body))
    out = frame.rename(
        columns={"TIME_PERIOD": "observation_date", "OBS_VALUE": "value"}
    ).copy()
    out["feature"] = out.REF_AREA.map(
        {country: name for name, country in countries.items()}
    )
    edition_month = pd.to_datetime(out.EDITION.astype(str), format="%Y%m")
    out["vintage_date"] = edition_month + pd.offsets.MonthEnd(0)
    out["available_date"] = edition_month + pd.offsets.MonthBegin(1)
    out["source"] = "OECD:LI_AA:first archived edition"
    # Monthly editions may already include within-month revisions. This is the
    # earliest archived edition, not a certified original publication value.
    return _first_observations(out)


def _fetch_macro_releases(
    start: str = "2012-01-01",
    end: str = "2023-01-01",
) -> pd.DataFrame:
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    seed = (first - pd.DateOffset(years=1)).replace(day=1).strftime("%Y-%m-%d")
    fred = _fetch_fred(MACRO_SYMBOLS, seed, end)
    cli = _fetch_cli(CLI_COUNTRIES, seed, end)
    releases = pd.concat([fred, cli], ignore_index=True)
    releases = (
        releases.loc[releases.available_date.lt(stop)]
        .sort_values(["feature", "observation_date"])
        .reset_index(drop=True)
    )
    return releases


def _align_yields(
    releases: pd.DataFrame, sessions: pd.DatetimeIndex, lag_days: int
) -> pd.DataFrame:
    """Use the last reported yield on or before session minus lag_days."""
    out = pd.DataFrame(index=sessions)
    observation_dates = sessions - pd.Timedelta(days=lag_days)
    for feature, group in releases.groupby("feature", sort=False):
        history = group.sort_values("observation_date").set_index("observation_date")
        values = []
        for session, observation_date in zip(sessions, observation_dates, strict=True):
            known = history.loc[history.available_date <= session, "value"]
            selected = known.reindex([observation_date], method="ffill")
            values.append(selected.iloc[0])
        out[feature] = values
    return out


def _align_cli(releases: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Carry the latest available CLI observation's first archived value."""
    out = pd.DataFrame(index=sessions)
    left = pd.DataFrame({"session": sessions.as_unit("ns")})
    for feature, group in releases.groupby("feature", sort=False):
        events = group.sort_values(["available_date", "observation_date"])
        events["available_date"] = events.available_date.astype("datetime64[ns]")
        events = events.drop_duplicates("available_date", keep="last")
        # A late publication of an older period must not roll the feature back.
        events = events.loc[
            events.observation_date.eq(events.observation_date.cummax())
        ]
        aligned = pd.merge_asof(
            left,
            events,
            left_on="session",
            right_on="available_date",
            direction="backward",
        )
        out[feature] = aligned.value.to_numpy()
    return out


def fetch_raw(
    start: str = "2012-01-01",
    end: str = "2023-01-01",
    yield_lag_days: int = 7,
) -> pd.DataFrame:
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    start, end = first.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")
    market = _fetch_yf(MARKET_SYMBOLS, start, end)
    market[["dxy", "USDCNY"]] = market[["dxy", "USDCNY"]].ffill()
    market = market.loc[
        (market.index >= first) & (market.index < stop) & market.copper.notna()
    ]
    releases = _fetch_macro_releases(start, end)
    yields = releases.loc[releases.feature.isin(MACRO_SYMBOLS)]
    cli = releases.loc[releases.feature.isin(CLI_COUNTRIES)]
    raw = market.join(_align_yields(yields, market.index, yield_lag_days))  # type: ignore
    raw = raw.join(_align_cli(cli, market.index))  # type: ignore
    raw.index.name = "date"
    raw = raw.loc[:, [*MARKET_SYMBOLS, *MACRO_SYMBOLS, *CLI_COUNTRIES]]
    return raw.rename(
        columns={
            "real_yield_10y": f"real_yield_10y_lag_{yield_lag_days}d",
            "cli_usa": "cli_usa_latest_available",
            "cli_china": "cli_china_latest_available",
        }
    )
