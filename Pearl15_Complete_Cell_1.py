import base64
import hashlib
import html
import io
import json
import math
import re
import warnings
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy.stats import beta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
RAW_HELPER_VERSION = "3.1"
OUTPUT_FAMILIES = {"P50", "P30", "T30", "VBHP"}
METHOD_REFERENCES = {
    "Gas turbine normalization": (
        "NASA/TM-2016-219147, Practical Techniques for"
        " Modeling Gas Turbine Engine Performance"
    ),
    "Validation": (
        "Expanding chronological flight blocks; future"
        " flights never select or fit earlier models"
    ),
}


def signal_family(name):
    normalized = re.sub("[^A-Z0-9]+", "_", str(name).upper()).strip("_")
    aliases = {
        "AMBIENT_PRESSURE": "P20",
        "AMBIENT_TEMPERATURE": "T20",
        "ALTITUDE": "ALT",
        "MACH": "MN",
        "MACH_NUMBER": "MN",
        "ENGINE_RPM": "RPM",
        "ENGINE_SPEED": "RPM",
    }
    for alias, family in aliases.items():
        if normalized == alias or normalized.startswith(alias + "_"):
            return family
    match = re.match(
        (
            "^(P_?20|P_?30|P_?50|T_?20|T_?30|TGT|OIP|O"
            "IT|TAT|ALT|NH|NL|RPM|EPR\\w*|VBHP|VBLP|FF"
            "|MN|PACK|CAI|WAI)(?:_|$)"
        ),
        normalized,
    )
    if not match:
        return normalized
    family = match.group(1).replace("_", "")
    return "EPR" if family.startswith("EPR") else family


def measurement_unit(column):
    name = re.sub("[^A-Z0-9]+", "_", str(column).upper()).strip("_")
    for suffix, unit in [
        ("DEGC", "degC"),
        ("CELSIUS", "degC"),
        ("DEGF", "degF"),
        ("KELVIN", "K"),
        ("KPA", "kPa"),
        ("PSIA", "psi"),
        ("PSI", "psi"),
        ("BAR", "bar"),
        ("RPM", "rpm"),
        ("IPS", "in/s"),
        ("PC", "%"),
        ("FT", "ft"),
        ("PA", "Pa"),
        ("K", "K"),
    ]:
        if name.endswith("_" + suffix):
            return unit
    return "unspecified"


def to_kelvin(values, column):
    unit = measurement_unit(column)
    array = np.asarray(values, dtype=float)
    if unit == "K":
        return array
    if unit == "degC":
        return array + 273.15
    if unit == "degF":
        return (array - 32.0) * (5.0 / 9.0) + 273.15
    raise ValueError(
        "".join(
            (
                "Temperature units are not explicit in ",
                "{}".format(column),
                "; choose a column with K, DEGC or DEGF units",
            ),
        ),
    )


def from_kelvin(values, column):
    unit = measurement_unit(column)
    if unit == "K":
        return np.asarray(values)
    if unit == "degC":
        return np.asarray(values) - 273.15
    if unit == "degF":
        return (np.asarray(values) - 273.15) * (9.0 / 5.0) + 32.0
    raise ValueError(
        "".join(
            (
                "Temperature units are not explicit in ",
                "{}".format(column),
            ),
        ),
    )


def pressure_scale(column):
    scales = {
        "psi": 6894.757293168,
        "Pa": 1.0,
        "kPa": 1000.0,
        "bar": 100000.0,
    }
    if measurement_unit(column) not in scales:
        raise ValueError(
            "".join(
                (
                    "Pressure units are not explicit in ",
                    "{}".format(column),
                    "; physics normalization cannot infer them",
                ),
            ),
        )
    return scales[measurement_unit(column)]


def validate_choices(config):
    inputs, targets = (config["inputs"], config["targets"])
    if not inputs or not targets:
        raise ValueError(
            (
                "Choose at least one operating input a"
                "nd one measured output."
            ),
        )
    if (
        len(inputs) != len(set(inputs))
        or len(targets) != len(set(targets))
    ):
        raise ValueError(
            "Repeated input or output columns are not allowed.",
        )
    if len(inputs) > 12 or len(targets) > 8:
        raise ValueError(
            (
                "Use no more than 12 operating inputs "
                "and 8 measured outputs."
            ),
        )
    target_families = {signal_family(column) for column in targets}
    leakage = [
        column
        for column in inputs
        if signal_family(column) in target_families
    ]
    if leakage:
        raise ValueError(
            (
                (
                    "An output, or another version of "
                    "that output, is also selected as "
                    "an input: "
                )
                + ", ".join(leakage)
            ),
        )
    forbidden = [
        column
        for column in inputs
        if signal_family(column) in OUTPUT_FAMILIES
    ]
    if forbidden:
        raise ValueError(
            (
                (
                    "Keep anomaly measurements out of "
                    "operating inputs; they can hide a"
                    "nother parameter's deterioration:"
                    " "
                )
                + ", ".join(forbidden)
            ),
        )
    families = [signal_family(column) for column in inputs]
    duplicated = sorted(
        {
            family
            for family in families
            if families.count(family) > 1
        },
    )
    if duplicated:
        raise ValueError(
            (
                (
                    "Choose one input version per meas"
                    "urement family, rather than dupli"
                    "cate ADC channels or RPM/% versio"
                    "ns: "
                )
                + ", ".join(duplicated)
            ),
        )
    if (
        "EPR" in families
        and (not config.get("epr_confirmed", False))
    ):
        raise ValueError(
            (
                "Leave EPR out until engineers confirm"
                " its meaning and that it is independe"
                "nt of the selected outputs."
            ),
        )
    if (
        not 0 < config["fit_fraction"] < 1
        or not 0 < config["cal_fraction"] < 1
    ):
        raise ValueError(
            (
                "Learning and reference percentages mu"
                "st both be positive."
            ),
        )
    if config["fit_fraction"] + config["cal_fraction"] > 0.9 + 1e-12:
        raise ValueError(
            (
                "Reserve at least 10% of the flights f"
                "or later monitoring."
            ),
        )
    if not 100 <= int(config["max_flights"]) <= 300:
        raise ValueError(
            "Maximum flights must be between 100 and 300.",
        )
    if not 0.5 < config["coverage"] < 1:
        raise ValueError(
            (
                "Choose an envelope setting strictly b"
                "etween 50% and 100%."
            ),
        )
    required, window = config["persistence"]
    if not 1 <= required <= window:
        raise ValueError(
            (
                "The required unusual-flight count mus"
                "t fit inside the warning window."
            ),
        )
    event_date = config.get("event_date")
    if event_date is None:
        raise ValueError("Choose the recorded event or review date.")
    lower, upper = (config.get("range_start"), config.get("range_end"))
    if lower and upper and (lower > upper):
        raise ValueError(
            (
                "The history start date must be on or "
                "before its end date."
            ),
        )
    if lower and lower >= event_date:
        raise ValueError("History must start before the event date.")
    if upper and upper >= event_date:
        raise ValueError(
            (
                "History must end before the event dat"
                "e; the complete event day is excluded"
                "."
            ),
        )
    if not str(config.get("engine", "")).strip():
        raise ValueError(
            (
                "Enter the engine serial number or sel"
                "ected engine identifier."
            ),
        )
    if (
        config.get("flight_column")
        and (
            config.get("flight_column")
            == config.get("sample_column")
        )
    ):
        raise ValueError(
            (
                "Flight start and snapshot time must u"
                "se different fields; multiple snapsho"
                "ts must not be counted as separate fl"
                "ights."
            ),
        )
    if config.get("physics_confirmed", False):
        p20 = next(
            (
                column
                for column in inputs
                if signal_family(column) == "P20"
            ),
            None,
        )
        t20 = next(
            (
                column
                for column in inputs
                if signal_family(column) == "T20"
            ),
            None,
        )
        if not p20 or not t20:
            raise ValueError(
                (
                    "Confirmed inlet normalization req"
                    "uires P20 and T20 among the input"
                    "s."
                ),
            )
        pressure_scale(p20)
        to_kelvin([1.0], t20)
        for target in targets:
            if signal_family(target) in {"P30", "P50"}:
                pressure_scale(target)
            elif signal_family(target) == "T30":
                to_kelvin([1.0], target)


def select_flights(all_starts, config):
    validate_choices(config)
    anchor = pd.Timestamp(config["event_date"])
    starts = (
        (
            (
                pd.DatetimeIndex(
                    pd.to_datetime(
                        list(all_starts),
                        errors="coerce",
                    ),
                )
            ).dropna()
        ).drop_duplicates()
    ).sort_values()
    if starts.tz is not None:
        raise ValueError(
            (
                "Use catalogue-local timestamp strings"
                " consistently; mixed timezone-aware t"
                "imestamps are not accepted."
            ),
        )
    starts = starts[starts < anchor]
    if not len(starts):
        raise ValueError(
            (
                "No recorded flights precede the event"
                " date for this engine."
            ),
        )
    lower = (
        pd.Timestamp(config["range_start"])
        if config.get("range_start")
        else None
    )
    upper = (
        pd.Timestamp(config["range_end"]) + pd.Timedelta(days=1)
        if config.get("range_end")
        else anchor
    )
    mask = starts < upper
    if lower is not None:
        mask &= starts >= lower
    chosen = starts[mask][-int(config["max_flights"]):]
    if len(chosen) < 100:
        raise ValueError(
            "".join(
                (
                    "Only ",
                    "{}".format(len(chosen)),
                    (
                        " distinct recorded flights ar"
                        "e available in this range. At"
                        " least 100 are required; exte"
                        "nd the range or choose anothe"
                        "r engine/event."
                    ),
                ),
            ),
        )
    fit_count = int(math.floor(len(chosen) * config["fit_fraction"]))
    cal_count = int(math.floor(len(chosen) * config["cal_fraction"]))
    if (
        fit_count < 20
        or cal_count < 15
        or len(chosen) - fit_count - cal_count < 10
    ):
        raise ValueError(
            (
                "The split leaves too few flights for "
                "learning, reference checks or monitor"
                "ing."
            ),
        )
    positions = {stamp: index for index, stamp in enumerate(starts)}
    stages = (
        ["Learning"] * fit_count + ["Reference"] * cal_count
        + ["Monitoring"] * (len(chosen) - fit_count - cal_count)
    )
    flights = pd.DataFrame(
        {
            "__FlightStart": chosen,
            "__WindowCycle": np.arange(len(chosen)),
            "__Stage": stages,
        },
    )
    flights["__LeadFlights"] = [len(starts) - positions[stamp] for stamp in chosen]
    flights["__PlotCycle"] = -flights["__LeadFlights"]
    return flights


def prepare_measurements(raw, flights, config):
    frame = raw.copy()
    required = ["__Phase", "__FlightStart", "__SnapshotTime"]
    if any((column not in frame for column in required)):
        raise ValueError(
            (
                "The extraction did not supply the req"
                "uired flight and snapshot fields."
            ),
        )
    for column in ["__FlightStart", "__SnapshotTime"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=required)
    anchor = pd.Timestamp(config["event_date"])
    frame = (
        frame.loc[(
            (
                (frame["__FlightStart"] < anchor)
                & (frame["__SnapshotTime"] < anchor)
            )
            & (frame["__SnapshotTime"] >= frame["__FlightStart"])
        )]
    ).copy()
    next_starts = pd.Series(
        flights["__FlightStart"].shift(-1).to_numpy(),
        index=flights["__FlightStart"],
    )
    next_time = frame["__FlightStart"].map(next_starts)
    overlap = next_time.notna() & (frame["__SnapshotTime"] >= next_time)
    overlap_count = int(overlap.sum())
    frame = frame.loc[~overlap].copy()
    measurements = [
        column
        for column in dict.fromkeys(config["inputs"] + config["targets"])
        if column in frame
    ]
    for column in measurements:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
    if "__AdditionalKey" not in frame:
        frame["__AdditionalKey"] = ""
    frame["__AdditionalKey"] = frame["__AdditionalKey"].fillna("").astype(str)
    keys = [
        "__Phase",
        "__FlightStart",
        "__SnapshotTime",
        "__AdditionalKey",
    ]
    exact_before = len(frame)
    frame = frame.drop_duplicates(subset=keys + measurements)
    exact_duplicates = exact_before - len(frame)
    conflicting = frame.duplicated(subset=keys, keep=False)
    conflict_rows = int(conflicting.sum())
    frame = frame.loc[~conflicting].copy()
    frame = frame.merge(
        flights,
        on="__FlightStart",
        how="inner",
        validate="many_to_one",
    )
    frame = (
        frame.sort_values(
            ["__WindowCycle", "__Phase", "__SnapshotTime"],
            kind="stable",
        )
    ).reset_index(
        drop=True,
    )
    diagnostics = {
        "Discarded invalid/date rows": before - exact_before,
        "Snapshots overlapping a later recorded flight": overlap_count,
        "Identical duplicate rows removed": exact_duplicates,
        "Conflicting snapshot rows excluded": conflict_rows,
        "Usable catalogue rows in selected flights": len(frame),
    }
    return (frame, diagnostics)


def physical_validity(values, column, config):
    data = pd.Series(values, copy=False)
    finite = data.notna() & np.isfinite(data.to_numpy(dtype=float))
    family, unit = (signal_family(column), measurement_unit(column))
    if (
        family in {"T20", "T30", "TAT", "TGT", "OIT"}
        and unit in {"degC", "degF", "K"}
    ):
        return (
            finite
            & pd.Series(
                to_kelvin(data, column) > 0,
                index=data.index,
            )
        )
    if family in {"NH", "NL", "RPM", "VBHP", "VBLP", "FF", "MN"}:
        return finite & data.ge(0)
    if (
        family in {"P20", "P30", "P50"}
        and config.get("physics_confirmed", False)
    ):
        return finite & data.gt(0)
    return finite


def physics_audit(frame, config):
    rows = []
    for phase in config["phases"]:
        subset = frame.loc[frame["__Phase"] == phase]
        for column in dict.fromkeys(config["inputs"] + config["targets"]):
            if column not in subset:
                continue
            invalid = (
                subset[column].notna()
                & ~physical_validity(subset[column], column, config)
            )
            rows.append(
                {
                    "Phase": phase,
                    "Parameter": column,
                    "Units": measurement_unit(column),
                    "Missing/nonfinite readings": int(subset[column].isna().sum()),
                    "Physical/sensor review readings": int(invalid.sum()),
                    "Monitoring physical/sensor review readings": int(
                        (
                            (
                                invalid
                                & subset["__Stage"].eq(
                                    "Monitoring",
                                )
                            )
                        ).sum(),
                    ),
                },
            )
    return pd.DataFrame(rows)


class EngineeringRegressor:

    def __init__(self, kind, inputs, target, physics_confirmed=False):
        self.kind, self.inputs, self.target = (kind, list(inputs), target)
        self.physics_confirmed = physics_confirmed
        self.p20 = next(
            (
                column
                for column in inputs
                if signal_family(column) == "P20"
            ),
            None,
        )
        self.t20 = next(
            (
                column
                for column in inputs
                if signal_family(column) == "T20"
            ),
            None,
        )

    def features(self, frame):
        result = frame[self.inputs].astype(float).copy()
        for column in self.inputs:
            if (
                signal_family(column) in {"T20", "T30", "TAT"}
                and measurement_unit(column) in {"K", "degC", "degF"}
            ):
                result[column] = to_kelvin(result[column], column)
        if self.physics_confirmed:
            theta = to_kelvin(frame[self.t20], self.t20) / 288.15
            for column in self.inputs:
                if signal_family(column) in {"NH", "NL", "RPM"}:
                    corrected = (
                        frame[column].to_numpy(dtype=float)
                        / np.sqrt(theta)
                    )
                    result[column + "_inlet_corrected"] = corrected
            result["inlet_delta"] = (
                (
                    frame[self.p20].to_numpy(dtype=float)
                    * pressure_scale(self.p20)
                )
                / 101325.0
            )
            result["inlet_theta"] = theta
        return result

    def transform_target(self, frame, values):
        if (
            self.physics_confirmed
            and signal_family(self.target) in {"P30", "P50"}
        ):
            return (
                np.asarray(values) * pressure_scale(self.target)
                / (
                    frame[self.p20].to_numpy(dtype=float)
                    * pressure_scale(self.p20)
                )
            )
        if (
            self.physics_confirmed
            and signal_family(self.target) == "T30"
        ):
            return (
                to_kelvin(values, self.target)
                / to_kelvin(frame[self.t20], self.t20)
            )
        return np.asarray(values, dtype=float)

    def inverse_target(self, frame, values):
        if (
            self.physics_confirmed
            and signal_family(self.target) in {"P30", "P50"}
        ):
            return (
                (
                    (
                        np.asarray(values)
                        * frame[self.p20].to_numpy(dtype=float)
                    )
                    * pressure_scale(self.p20)
                )
                / pressure_scale(self.target)
            )
        if (
            self.physics_confirmed
            and signal_family(self.target) == "T30"
        ):
            return from_kelvin(
                (
                    np.asarray(values)
                    * to_kelvin(frame[self.t20], self.t20)
                ),
                self.target,
            )
        return np.asarray(values, dtype=float)

    def fit(self, frame, values, sample_weight=None):
        transformed = self.transform_target(frame, values)
        if self.kind == "Median baseline":
            order = np.argsort(transformed)
            weights = (
                np.ones(len(transformed))
                if sample_weight is None
                else np.asarray(sample_weight)
            )
            cumulative = np.cumsum(weights[order])
            self.constant = float(
                transformed[order[min(
                    np.searchsorted(
                        cumulative,
                        cumulative[-1] / 2.0,
                    ),
                    len(order) - 1,
                )]],
            )
        else:
            model = (
                Ridge(alpha=1.0, solver="svd")
                if self.kind == "Ridge"
                else RandomForestRegressor(
                    n_estimators=128,
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=1,
                )
            )
            self.estimator = Pipeline(
                [("scale", StandardScaler()), ("model", model)],
            )
            self.estimator.fit(
                self.features(frame),
                transformed,
                scale__sample_weight=sample_weight,
                model__sample_weight=sample_weight,
            )
        return self

    def predict(self, frame):
        predictions = (
            np.repeat(self.constant, len(frame))
            if self.kind == "Median baseline"
            else self.estimator.predict(self.features(frame))
        )
        return self.inverse_target(frame, predictions)


def make_estimator(kind, inputs=None, target=None, physics_confirmed=False):
    if kind not in {"Ridge", "Random forest", "Median baseline"}:
        raise ValueError("Unrecognized prediction model: " + str(kind))
    return EngineeringRegressor(
        kind,
        inputs or [],
        target,
        physics_confirmed,
    )


def fit_estimator(estimator, frame, inputs, target):
    counts = (
        frame.groupby("__WindowCycle")["__WindowCycle"].transform(
            "size",
        )
    ).to_numpy(
        dtype=float,
    )
    weights = 1.0 / counts
    weights *= len(weights) / weights.sum()
    return estimator.fit(
        frame[inputs],
        frame[target],
        sample_weight=weights,
    )


def per_flight_mae(frame, target, predictions):
    errors = pd.Series(
        np.abs(frame[target].to_numpy(dtype=float) - predictions),
        index=frame.index,
    )
    return float(errors.groupby(frame["__WindowCycle"]).mean().mean())


def choose_estimator(train, inputs, target, mode, physics_confirmed=False):
    cycles = np.sort(train["__WindowCycle"].unique())
    first_end = max(15, int(len(cycles) * 0.5))
    boundaries = np.unique(np.linspace(first_end, len(cycles), 4, dtype=int))
    kinds = (
        ["Ridge", "Random forest", "Median baseline"]
        if mode == "Auto"
        else list(dict.fromkeys([mode, "Median baseline"]))
    )
    scores = {kind: [] for kind in kinds}
    for begin, finish in zip(boundaries[:-1], boundaries[1:]):
        earlier = train.loc[train["__WindowCycle"].isin(cycles[:begin])]
        later = train.loc[train["__WindowCycle"].isin(cycles[begin:finish])]
        if later["__WindowCycle"].nunique() < 3:
            continue
        for kind in kinds:
            estimator = fit_estimator(
                make_estimator(
                    kind,
                    inputs,
                    target,
                    physics_confirmed,
                ),
                earlier,
                inputs,
                target,
            )
            scores[kind].append(
                per_flight_mae(
                    later,
                    target,
                    estimator.predict(later[inputs]),
                ),
            )
    averages = {
        kind: float(np.mean(values))
        for kind, values in scores.items()
        if values
    }
    if not averages:
        return (
            "Ridge" if mode == "Auto" else mode,
            {"Selection": "Too few earlier validation flights"},
        )
    winner = mode if mode != "Auto" else min(averages, key=averages.get)
    if (
        mode == "Auto"
        and winner == "Random forest"
        and (averages["Random forest"] >= 0.95 * averages["Ridge"])
    ):
        winner = "Ridge"
    return (winner, averages)


def build_operating_domain(train, inputs):
    center = train[inputs].median().to_numpy(dtype=float)
    spread = (
        (
            train[inputs].quantile(0.75)
            - train[inputs].quantile(0.25)
        )
    ).to_numpy(
        dtype=float,
    )
    ranges = (
        train[inputs].max().to_numpy(dtype=float)
        - train[inputs].min().to_numpy(dtype=float)
    )
    scale = np.maximum(
        spread,
        np.maximum(
            ranges * 0.1,
            np.maximum(np.abs(center) * 1e-06, 1e-09),
        ),
    )
    standardized = (train[inputs].to_numpy(dtype=float) - center) / scale
    neighbors = NearestNeighbors(n_neighbors=min(4, len(train))).fit(
        standardized,
    )
    distances = neighbors.kneighbors(standardized)[0][:, -1]
    threshold = max(float(np.quantile(distances, 0.99)) * 1.5, 0.05)
    return {
        "center": center,
        "scale": scale,
        "neighbors": neighbors,
        "distance_threshold": threshold,
        "bounds": {
            column: [
                float(train[column].min()),
                float(train[column].max()),
            ]
            for column in inputs
        },
    }


def operating_domain(frame, train, inputs, domain=None):
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    domain = domain or build_operating_domain(train, inputs)
    valid = pd.Series(True, index=frame.index)
    for column in inputs:
        lower, upper = domain["bounds"][column]
        slack = max((upper - lower) * 0.1, abs(lower) * 1e-06, 1e-09)
        valid &= frame[column].between(lower - slack, upper + slack)
    standardized = (
        (frame[inputs].to_numpy(dtype=float) - domain["center"])
        / domain["scale"]
    )
    distances = domain["neighbors"].kneighbors(standardized)[0][:, -1]
    return (
        valid
        & pd.Series(
            distances <= domain["distance_threshold"],
            index=frame.index,
        )
    )


def empirical_band(reference, target, predictions, coverage, offset=None):
    residual_series = pd.Series(
        reference[target].to_numpy(dtype=float) - predictions,
        index=reference.index,
    )
    flight_medians = (
        residual_series.groupby(
            reference["__WindowCycle"],
            sort=True,
        )
    ).median()
    offset = (
        float(flight_medians.median())
        if offset is None
        else float(offset)
    )
    centered = residual_series - offset
    maxima = np.sort(
        (
            (
                centered.abs().groupby(
                    reference["__WindowCycle"],
                )
            ).max()
        ).to_numpy(
            dtype=float,
        ),
    )
    rank = min(
        len(maxima),
        max(1, int(np.ceil((len(maxima) + 1) * coverage))),
    )
    quantile = float(maxima[rank - 1])
    noise = float(
        (
            1.4826
            * np.median(
                np.abs(
                    (
                        flight_medians.to_numpy(dtype=float)
                        - flight_medians.median()
                    ),
                ),
            )
        ),
    )
    floor = max(abs(float(reference[target].median())) * 1e-06, 1e-09)
    band = max(quantile, 3.0 * noise, floor)
    half = max(1, len(flight_medians) // 2)
    shift = abs(
        (
            float(flight_medians.iloc[:half].median())
            - float(flight_medians.iloc[-half:].median())
        ),
    )
    differences = np.diff(flight_medians.to_numpy(dtype=float))
    within_noise = (
        float(
            (
                (
                    1.4826
                    * np.median(
                        np.abs(
                            differences - np.median(differences),
                        ),
                    )
                )
                / np.sqrt(2.0)
            ),
        )
        if len(differences)
        else noise
    )
    stable = shift <= max(3.0 * within_noise, 0.25 * band)
    return (offset, band, stable, shift)


def persistent_flags(values, required, window):
    return (
        (
            values.rolling(window=window, min_periods=window).sum()
            >= required
        )
        & values.eq(1.0)
    )


def analyse_measurements(frame, flights, config):
    validate_choices(config)
    records, metrics, summaries, failures, models = ([], [], [], [], {})
    monitoring_cycles = (
        flights.loc[flights["__Stage"] == "Monitoring", "__WindowCycle"]
    ).to_numpy()
    selected_reference = int(flights["__Stage"].eq("Reference").sum())
    audit = physics_audit(frame, config)
    missing_targets = [
        target
        for target in config["targets"]
        if target not in frame or frame[target].notna().sum() == 0
    ]
    for phase in config["phases"]:
        phase_frame = frame.loc[frame["__Phase"] == phase].copy()
        for target in config["targets"]:
            label = {"Phase": phase, "Parameter": target}
            try:
                required_columns = config["inputs"] + [target]
                absent = [
                    column
                    for column in required_columns
                    if column not in phase_frame
                ]
                if absent:
                    raise ValueError(
                        (
                            "Columns unavailable: "
                            + ", ".join(absent)
                        ),
                    )
                mask = pd.Series(True, index=phase_frame.index)
                for column in required_columns:
                    mask &= physical_validity(
                        phase_frame[column],
                        column,
                        config,
                    )
                valid = phase_frame.loc[mask].copy()
                train = valid.loc[valid["__Stage"] == "Learning"].copy()
                reference_all = valid.loc[valid["__Stage"] == "Reference"].copy()
                later = valid.loc[valid["__Stage"] == "Monitoring"].copy()
                train_count = train["__WindowCycle"].nunique()
                if train_count < 20:
                    raise ValueError(
                        "".join(
                            (
                                "Only ",
                                "{}".format(train_count),
                                (
                                    " learning flights"
                                    " have complete, p"
                                    "hysically admissi"
                                    "ble readings; at "
                                    "least 20 are requ"
                                    "ired"
                                ),
                            ),
                        ),
                    )
                confirmed_physics = config.get("physics_confirmed", False)
                kind, cv = choose_estimator(
                    train,
                    config["inputs"],
                    target,
                    config["model_mode"],
                    confirmed_physics,
                )
                estimator = fit_estimator(
                    make_estimator(
                        kind,
                        config["inputs"],
                        target,
                        confirmed_physics,
                    ),
                    train,
                    config["inputs"],
                    target,
                )
                domain = build_operating_domain(train, config["inputs"])
                reference = (
                    reference_all.loc[operating_domain(
                        reference_all,
                        train,
                        config["inputs"],
                        domain,
                    )]
                ).copy()
                ref_count = reference["__WindowCycle"].nunique()
                if ref_count < 15:
                    raise ValueError(
                        "".join(
                            (
                                "Only ",
                                "{}".format(ref_count),
                                (
                                    " reference flight"
                                    "s have complete r"
                                    "eadings in compar"
                                    "able operating co"
                                    "nditions; at leas"
                                    "t 15 are required"
                                ),
                            ),
                        ),
                    )
                if later["__WindowCycle"].nunique() < 10:
                    raise ValueError(
                        (
                            "Fewer than 10 monitoring "
                            "flights have complete, ph"
                            "ysically admissible readi"
                            "ngs for this output"
                        ),
                    )
                train_residuals = pd.Series(
                    (
                        train[target].to_numpy(dtype=float)
                        - estimator.predict(
                            train[config["inputs"]],
                        )
                    ),
                    index=train.index,
                )
                train_bias = float(
                    (
                        (
                            train_residuals.groupby(
                                train["__WindowCycle"],
                            )
                        ).median()
                    ).median(),
                )
                ref_predictions = estimator.predict(reference[config["inputs"]])
                offset, band, stable, shift = empirical_band(
                    reference,
                    target,
                    ref_predictions,
                    config["coverage"],
                    offset=train_bias,
                )
                ref_residuals = pd.Series(
                    (
                        reference[target].to_numpy(dtype=float)
                        - (ref_predictions + offset)
                    ),
                    index=reference.index,
                )
                ref_medians = (
                    ref_residuals.groupby(
                        reference["__WindowCycle"],
                    )
                ).median()
                reference_bias = float(ref_medians.median())
                bias_noise = float(
                    (
                        1.4826
                        * np.median(
                            np.abs(ref_medians - reference_bias),
                        )
                    ),
                )
                stable = bool(
                    (
                        stable
                        and (
                            abs(reference_bias)
                            <= max(3.0 * bias_noise, 0.5 * band)
                        )
                    ),
                )
                chosen_cv = cv.get(kind)
                median_cv = cv.get("Median baseline")
                skill_ok = (
                    chosen_cv is not None
                    and median_cv is not None
                    and (chosen_cv <= 1.1 * max(median_cv, 1e-09))
                )
                reference_fraction = ref_count / selected_reference
                review_reasons = []
                if not stable:
                    review_reasons.append(
                        "Learning/reference residual shift",
                    )
                if not skill_ok:
                    review_reasons.append(
                        (
                            "Insufficient earlier pred"
                            "iction validation"
                        ),
                    )
                if reference_fraction < 0.7:
                    review_reasons.append(
                        (
                            "Less than 70% of referenc"
                            "e flights assessable"
                        ),
                    )
                if (
                    train[target].nunique() < 2
                    and reference[target].nunique() < 2
                ):
                    review_reasons.append(
                        (
                            "Constant output channel; "
                            "sensor resolution/validit"
                            "y needs review"
                        ),
                    )
                baseline_ok = not review_reasons
                predicted = (
                    estimator.predict(valid[config["inputs"]])
                    + offset
                )
                result = (
                    valid[[
                        "__Phase",
                        "__FlightStart",
                        "__SnapshotTime",
                        "__WindowCycle",
                        "__Stage",
                        "__LeadFlights",
                        "__PlotCycle",
                    ]]
                ).copy()
                (
                    result["Parameter"],
                    result["Observed"],
                    result["Expected"],
                ) = (
                    target,
                    valid[target].to_numpy(dtype=float),
                    predicted,
                )
                result["Lower"], result["Upper"] = (predicted - band, predicted + band)
                result["Residual"] = result["Observed"] - predicted
                result["Score"] = result["Residual"] / band
                result["In learned conditions"] = (
                    operating_domain(
                        valid,
                        train,
                        config["inputs"],
                        domain,
                    )
                ).to_numpy()
                result["Prediction physically admissible"] = (
                    physical_validity(
                        pd.Series(predicted, index=valid.index),
                        target,
                        config,
                    )
                ).to_numpy()
                result["Baseline/model accepted"] = baseline_ok
                comparable = (
                    result["In learned conditions"]
                    & result["Prediction physically admissible"]
                )
                result["Unusual reading"] = (
                    (
                        result["__Stage"].eq("Monitoring")
                        & comparable
                    )
                    & result["Score"].abs().gt(1.0)
                )
                result["Persistent deviation"] = False
                result["Persistent warning"] = False
                result["Warning direction"] = ""
                eligible = result.loc[result["__Stage"].eq("Monitoring") & comparable]
                directions = {}
                for direction, condition in [
                    ("Above expected", eligible["Score"].gt(1.0)),
                    (
                        "Below expected",
                        eligible["Score"].lt(-1.0),
                    ),
                ]:
                    values = pd.Series(
                        np.nan,
                        index=monitoring_cycles,
                        dtype=float,
                    )
                    flags = (
                        (
                            condition.groupby(
                                eligible["__WindowCycle"],
                            )
                        ).max()
                    ).astype(
                        float,
                    )
                    values.loc[flags.index] = flags
                    persisted = persistent_flags(
                        values,
                        *config["persistence"],
                    )
                    cycles = persisted.index[persisted].tolist()
                    directions[direction] = cycles
                    points = (
                        (
                            result["Unusual reading"]
                            & result["__WindowCycle"].isin(cycles)
                        )
                        & (
                            result["Score"].gt(0)
                            if direction == "Above expected"
                            else result["Score"].lt(0)
                        )
                    )
                    result.loc[points, "Persistent deviation"] = True
                    result.loc[points, "Warning direction"] = direction
                    result.loc[points, "Persistent warning"] = baseline_ok
                unusual = result.loc[result["Unusual reading"]].sort_values(
                    ["__WindowCycle", "__SnapshotTime"],
                )
                warnings = (
                    result.loc[result["Persistent warning"]]
                ).sort_values(
                    ["__WindowCycle", "__SnapshotTime"],
                )
                first = warnings.iloc[0] if not warnings.empty else None
                assessed = eligible["__WindowCycle"].nunique()
                monitoring_fraction = assessed / len(monitoring_cycles)
                unscored = len(monitoring_cycles) - assessed
                if not baseline_ok:
                    status = (
                        "Review baseline/model; anomal"
                        "y not established"
                    )
                elif first is not None:
                    status = "Persistent statistical anomaly"
                elif not unusual.empty:
                    status = "Isolated statistical deviations"
                elif monitoring_fraction < 0.8:
                    status = (
                        "Incomplete monitoring; no ano"
                        "maly in assessed flights"
                    )
                else:
                    status = "No statistical anomaly in assessed flights"
                warning_cycles = (
                    result.loc[result["Persistent warning"], "__WindowCycle"]
                ).nunique()
                summaries.append(
                    {
                        **label,
                        "Status": status,
                        "Direction": (
                            first["Warning direction"]
                            if first is not None
                            else ""
                        ),
                        "First unusual flight": (
                            unusual.iloc[0]["__FlightStart"]
                            if not unusual.empty
                            else pd.NaT
                        ),
                        "First persistent warning": (
                            first["__FlightStart"]
                            if first is not None
                            else pd.NaT
                        ),
                        "Warning snapshot time": (
                            first["__SnapshotTime"]
                            if first is not None
                            else pd.NaT
                        ),
                        "Recorded flights before event": (
                            int(first["__LeadFlights"])
                            if first is not None
                            else None
                        ),
                        "Days before event date": (
                            (
                                (
                                    (
                                        pd.Timestamp(
                                            config["event_date"],
                                        )
                                    ).normalize()
                                    - (
                                        pd.Timestamp(
                                            first["__SnapshotTime"],
                                        )
                                    ).normalize()
                                )
                            ).days
                            if first is not None
                            else None
                        ),
                        "Unusual monitoring flights": unusual["__WindowCycle"].nunique(),
                        "Persistent warning flights": warning_cycles,
                        "Assessable monitoring flights": assessed,
                        "Unassessed monitoring flights": unscored,
                        "Monitoring coverage %": round(100 * monitoring_fraction, 1),
                        "Reference stability": (
                            "No large shift found"
                            if stable
                            else "Review baseline"
                        ),
                        "Review reason": "; ".join(review_reasons),
                        "Healthy baseline": (
                            "User confirmed against records"
                            if config.get(
                                "baseline_confirmed",
                                False,
                            )
                            else "Assumed; not verified"
                        ),
                    },
                )
                metrics.append(
                    {
                        **label,
                        "Model": kind,
                        "Physics normalization": (
                            (
                                "Confirmed inlet ratio"
                                "s and corrected speed"
                            )
                            if confirmed_physics
                            else (
                                "Measured operating in"
                                "puts; inlet meaning u"
                                "nconfirmed"
                            )
                        ),
                        "Learning flights": train_count,
                        "Reference flights": ref_count,
                        "Monitoring flights with readings": later["__WindowCycle"].nunique(),
                        (
                            "Monitoring flights in lea"
                            "rned conditions"
                        ): assessed,
                        "Earlier validation MAE": chosen_cv,
                        "Earlier median-baseline MAE": median_cv,
                        "Reference MAE": per_flight_mae(
                            reference,
                            target,
                            ref_predictions + offset,
                        ),
                        "Envelope setting %": 100 * config["coverage"],
                        "Reference band half-width": band,
                        "Reference shift": shift,
                        "Learning-to-reference residual bias": reference_bias,
                        "Rows missing or physically inadmissible": len(phase_frame) - len(valid),
                        "Reference note": (
                            "Empirical flight-level en"
                            "velope; small reference s"
                            "ets can give identical 95"
                            "%/99% bands; no guarantee"
                            "d false-alarm probability"
                        ),
                    },
                )
                models[phase, target] = {
                    "estimator": estimator,
                    "offset": offset,
                    "band": band,
                    "inputs": list(config["inputs"]),
                    "selection_scores": cv,
                    "training_flights": sorted(
                        train["__WindowCycle"].unique().tolist(),
                    ),
                    "reference_flights": sorted(
                        (
                            reference["__WindowCycle"].unique()
                        ).tolist(),
                    ),
                    "training_bounds": domain["bounds"],
                    "domain": domain,
                    "baseline_accepted": baseline_ok,
                    "review_reasons": review_reasons,
                }
                records.append(result)
            except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
                failures.append({**label, "Reason": str(exc)})
    readings = (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame()
    )
    return {
        "readings": readings,
        "summary": pd.DataFrame(summaries),
        "metrics": pd.DataFrame(metrics),
        "unavailable": pd.DataFrame(failures),
        "models": models,
        "flights": flights,
        "config": dict(config),
        "missing_targets": missing_targets,
        "physics_checks": audit,
    }


EVENTS = [
    {
        "esn": "56319",
        "date": "2026-09-07",
        "kind": "Other TRU event",
        "aircraft": "60175",
        "title": "OSD72952 - A/C 60175 - MULTIPLE TRU FAULTS-TRU",
        "id": "event_000",
        "label": "ESN 56319 \u00b7 2026-09-07 \u00b7 Other TRU event",
    },
    {
        "esn": "56149",
        "date": "2026-07-19",
        "kind": "HPT2 blade-off",
        "aircraft": "60080",
        "id": "event_001",
        "label": "ESN 56149 \u00b7 2026-07-19 \u00b7 HPT2 blade-off",
    },
    {
        "esn": "56254",
        "date": "2026-06-08",
        "kind": "Other TRU event",
        "aircraft": "60135",
        "title": (
            "OSD71646 - A/C 60135 - RH TRU | R REV LOC"
            "K FAULT (Repeat) - ABTO - TRU (prev OSD 7"
            "0608)"
        ),
        "id": "event_002",
        "label": "ESN 56254 \u00b7 2026-06-08 \u00b7 Other TRU event",
    },
    {
        "esn": "56319",
        "date": "2026-05-15",
        "kind": "Other TRU event",
        "aircraft": "60175",
        "title": (
            "OSD71292 - A/C 60175 - LH REVERSER FAIL ("
            "amber) CAS message displayed - TRU"
        ),
        "id": "event_003",
        "label": "ESN 56319 \u00b7 2026-05-15 \u00b7 Other TRU event",
    },
    {
        "esn": "56319",
        "date": "2026-05-07",
        "kind": "Other TRU event",
        "aircraft": "60175",
        "title": (
            "OSD71185 - A/C 60175 - L Reverser Fail & "
            "L FADEC Fault CAS Messages - TRU"
        ),
        "id": "event_004",
        "label": "ESN 56319 \u00b7 2026-05-07 \u00b7 Other TRU event",
    },
    {
        "esn": "56319",
        "date": "2026-04-25",
        "kind": "Other TRU event",
        "aircraft": "60175",
        "title": (
            "OSD71027 - A/C 60175 - L REVERSER FAIL - "
            "L FADEC FAULT - TRU"
        ),
        "id": "event_005",
        "label": "ESN 56319 \u00b7 2026-04-25 \u00b7 Other TRU event",
    },
    {
        "esn": "56205",
        "date": "2026-04-13",
        "kind": "Other TRU event",
        "aircraft": "60112",
        "title": (
            "OSD70807 - A/C 60112 | LEFT TRU CRACKED S"
            "UPPORT _ CRACKED RAMP FAIRING - TRU"
        ),
        "id": "event_006",
        "label": "ESN 56205 \u00b7 2026-04-13 \u00b7 Other TRU event",
    },
    {
        "esn": "56254",
        "date": "2026-03-28",
        "kind": "Other TRU event",
        "aircraft": "60135",
        "title": (
            "OSD70608 - A/C 60135 - RH TRU | R REV LOC"
            "K FAULT (Repeat) - ABTO - TRU"
        ),
        "id": "event_007",
        "label": "ESN 56254 \u00b7 2026-03-28 \u00b7 Other TRU event",
    },
    {
        "esn": "56036",
        "date": "2026-03-28",
        "kind": "HPT1 inspection",
        "score": 5,
        "id": "event_008",
        "label": (
            "ESN 56036 \u00b7 2026-03-28 \u00b7 HPT1 i"
            "nspection; score 5"
        ),
    },
    {
        "esn": "56017",
        "date": "2026-03-28",
        "kind": "HPT1 inspection",
        "score": 5,
        "id": "event_009",
        "label": (
            "ESN 56017 \u00b7 2026-03-28 \u00b7 HPT1 i"
            "nspection; score 5"
        ),
    },
    {
        "esn": "56084",
        "date": "2026-03-22",
        "kind": "HPT1 inspection",
        "score": 6,
        "id": "event_010",
        "label": (
            "ESN 56084 \u00b7 2026-03-22 \u00b7 HPT1 i"
            "nspection; score 6"
        ),
    },
    {
        "esn": "56083",
        "date": "2026-03-22",
        "kind": "HPT1 inspection",
        "score": 5,
        "id": "event_011",
        "label": (
            "ESN 56083 \u00b7 2026-03-22 \u00b7 HPT1 i"
            "nspection; score 5"
        ),
    },
    {
        "esn": "56190",
        "date": "2026-03-11",
        "kind": "HPT2 BSI removal",
        "aircraft": "60085",
        "id": "event_012",
        "label": "ESN 56190 \u00b7 2026-03-11 \u00b7 HPT2 BSI removal",
    },
    {
        "esn": "56007",
        "date": "2026-03-10",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_013",
        "label": (
            "ESN 56007 \u00b7 2026-03-10 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56172",
        "date": "2026-02-25",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_014",
        "label": (
            "ESN 56172 \u00b7 2026-02-25 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56171",
        "date": "2026-02-25",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_015",
        "label": (
            "ESN 56171 \u00b7 2026-02-25 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56006",
        "date": "2026-02-22",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_016",
        "label": (
            "ESN 56006 \u00b7 2026-02-22 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56197",
        "date": "2026-02-21",
        "kind": "Other TRU event",
        "aircraft": "60097",
        "title": (
            "OSD70141-A/C60097-SPARES RRD-TERTIARY LOC"
            "K,PN P516A0001-00,1XEA+DOOR SEAL REPLACEM"
            "ENT,PHENIX JET-TRU"
        ),
        "id": "event_017",
        "label": "ESN 56197 \u00b7 2026-02-21 \u00b7 Other TRU event",
    },
    {
        "esn": "56165",
        "date": "2026-02-16",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_018",
        "label": (
            "ESN 56165 \u00b7 2026-02-16 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56136",
        "date": "2026-02-13",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_019",
        "label": (
            "ESN 56136 \u00b7 2026-02-13 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56135",
        "date": "2026-02-12",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_020",
        "label": (
            "ESN 56135 \u00b7 2026-02-12 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56342",
        "date": "2026-02-10",
        "kind": "Other TRU event",
        "aircraft": "60185",
        "title": "OSD70026 - A/C 60185 - R REVERSER FAIL Amber CAS - TRU",
        "id": "event_021",
        "label": "ESN 56342 \u00b7 2026-02-10 \u00b7 Other TRU event",
    },
    {
        "esn": "56191",
        "date": "2026-02-09",
        "kind": "Other TRU event",
        "aircraft": "60090",
        "title": "Bombardier - L Reverser Fail (aborted takeoff) - TRU",
        "id": "event_022",
        "label": "ESN 56191 \u00b7 2026-02-09 \u00b7 Other TRU event",
    },
    {
        "esn": "56078",
        "date": "2026-01-14",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_023",
        "label": (
            "ESN 56078 \u00b7 2026-01-14 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56077",
        "date": "2026-01-13",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_024",
        "label": (
            "ESN 56077 \u00b7 2026-01-13 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56013",
        "date": "2026-01-08",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_025",
        "label": (
            "ESN 56013 \u00b7 2026-01-08 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56012",
        "date": "2026-01-08",
        "kind": "HPT1 inspection",
        "score": 5,
        "id": "event_026",
        "label": (
            "ESN 56012 \u00b7 2026-01-08 \u00b7 HPT1 i"
            "nspection; score 5"
        ),
    },
    {
        "esn": "56050",
        "date": "2025-12-16",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_027",
        "label": (
            "ESN 56050 \u00b7 2025-12-16 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56016",
        "date": "2025-12-05",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_028",
        "label": (
            "ESN 56016 \u00b7 2025-12-05 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56009",
        "date": "2025-12-05",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_029",
        "label": (
            "ESN 56009 \u00b7 2025-12-05 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56131",
        "date": "2025-11-05",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_030",
        "label": (
            "ESN 56131 \u00b7 2025-11-05 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56186",
        "date": "2025-10-30",
        "kind": "HPT1 inspection",
        "score": 6,
        "id": "event_031",
        "label": (
            "ESN 56186 \u00b7 2025-10-30 \u00b7 HPT1 i"
            "nspection; score 6"
        ),
    },
    {
        "esn": "56185",
        "date": "2025-10-30",
        "kind": "HPT1 inspection",
        "score": 6,
        "id": "event_032",
        "label": (
            "ESN 56185 \u00b7 2025-10-30 \u00b7 HPT1 i"
            "nspection; score 6"
        ),
    },
    {
        "esn": "56067",
        "date": "2025-10-14",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_033",
        "label": (
            "ESN 56067 \u00b7 2025-10-14 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56050",
        "date": "2025-10-13",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_034",
        "label": (
            "ESN 56050 \u00b7 2025-10-13 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56283",
        "date": "2025-09-30",
        "kind": "Other TRU event",
        "aircraft": "60153",
        "title": (
            "OSD68284 - A/C 60153 - ABORTED TAKE-OFF -"
            " L REVERSER FAIL AND L FADEC FAIL CAS MES"
            "SAGES - TRU"
        ),
        "id": "event_035",
        "label": "ESN 56283 \u00b7 2025-09-30 \u00b7 Other TRU event",
    },
    {
        "esn": "56110",
        "date": "2025-09-23",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_036",
        "label": (
            "ESN 56110 \u00b7 2025-09-23 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56109",
        "date": "2025-09-23",
        "kind": "HPT1 inspection",
        "score": 3,
        "id": "event_037",
        "label": (
            "ESN 56109 \u00b7 2025-09-23 \u00b7 HPT1 i"
            "nspection; score 3"
        ),
    },
    {
        "esn": "56150",
        "date": "2025-09-08",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_038",
        "label": (
            "ESN 56150 \u00b7 2025-09-08 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56149",
        "date": "2025-09-08",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_039",
        "label": (
            "ESN 56149 \u00b7 2025-09-08 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56124",
        "date": "2025-08-29",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_040",
        "label": (
            "ESN 56124 \u00b7 2025-08-29 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56123",
        "date": "2025-08-28",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_041",
        "label": (
            "ESN 56123 \u00b7 2025-08-28 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56016",
        "date": "2025-08-26",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_042",
        "label": (
            "ESN 56016 \u00b7 2025-08-26 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56090",
        "date": "2025-06-20",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_043",
        "label": (
            "ESN 56090 \u00b7 2025-06-20 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56089",
        "date": "2025-06-20",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_044",
        "label": (
            "ESN 56089 \u00b7 2025-06-20 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56086",
        "date": "2025-05-30",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_045",
        "label": (
            "ESN 56086 \u00b7 2025-05-30 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56085",
        "date": "2025-05-30",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_046",
        "label": (
            "ESN 56085 \u00b7 2025-05-30 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56186",
        "date": "2025-05-17",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_047",
        "label": (
            "ESN 56186 \u00b7 2025-05-17 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56185",
        "date": "2025-05-17",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_048",
        "label": (
            "ESN 56185 \u00b7 2025-05-17 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56179",
        "date": "2025-05-08",
        "kind": "HPT1 inspection",
        "score": 5,
        "id": "event_049",
        "label": (
            "ESN 56179 \u00b7 2025-05-08 \u00b7 HPT1 i"
            "nspection; score 5"
        ),
    },
    {
        "esn": "56084",
        "date": "2025-05-02",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_050",
        "label": (
            "ESN 56084 \u00b7 2025-05-02 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56083",
        "date": "2025-05-02",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_051",
        "label": (
            "ESN 56083 \u00b7 2025-05-02 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56035",
        "date": "2025-04-01",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_052",
        "label": (
            "ESN 56035 \u00b7 2025-04-01 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56034",
        "date": "2025-04-01",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_053",
        "label": (
            "ESN 56034 \u00b7 2025-04-01 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56006",
        "date": "2025-03-27",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_054",
        "label": (
            "ESN 56006 \u00b7 2025-03-27 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56186",
        "date": "2025-03-08",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_055",
        "label": (
            "ESN 56186 \u00b7 2025-03-08 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56185",
        "date": "2025-03-08",
        "kind": "HPT1 inspection",
        "score": 4,
        "id": "event_056",
        "label": (
            "ESN 56185 \u00b7 2025-03-08 \u00b7 HPT1 i"
            "nspection; score 4"
        ),
    },
    {
        "esn": "56245",
        "date": "2025-03-07",
        "kind": "Other TRU event",
        "aircraft": "60133",
        "title": (
            "OSD64153 - A/C 60133 - SPARES - P/N T7013"
            "-3 (TRU STOW SWITCH) - TRU"
        ),
        "id": "event_057",
        "label": "ESN 56245 \u00b7 2025-03-07 \u00b7 Other TRU event",
    },
    {
        "esn": "56098",
        "date": "2025-02-12",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_058",
        "label": (
            "ESN 56098 \u00b7 2025-02-12 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56097",
        "date": "2025-02-12",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_059",
        "label": (
            "ESN 56097 \u00b7 2025-02-12 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56180",
        "date": "2024-12-23",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_060",
        "label": (
            "ESN 56180 \u00b7 2024-12-23 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56179",
        "date": "2024-12-23",
        "kind": "HPT1 inspection",
        "score": 2,
        "id": "event_061",
        "label": (
            "ESN 56179 \u00b7 2024-12-23 \u00b7 HPT1 i"
            "nspection; score 2"
        ),
    },
    {
        "esn": "56179",
        "date": "2024-12-12",
        "kind": "TRU leakage",
        "aircraft": "60098",
        "position": "LH",
        "osd": "63081",
        "id": "event_062",
        "label": "ESN 56179 \u00b7 2024-12-12 \u00b7 TRU leakage",
    },
    {
        "esn": "56186",
        "date": "2024-11-17",
        "kind": "HPT1 inspection",
        "score": 3,
        "id": "event_063",
        "label": (
            "ESN 56186 \u00b7 2024-11-17 \u00b7 HPT1 i"
            "nspection; score 3"
        ),
    },
    {
        "esn": "56185",
        "date": "2024-11-16",
        "kind": "HPT1 inspection",
        "score": 3,
        "id": "event_064",
        "label": (
            "ESN 56185 \u00b7 2024-11-16 \u00b7 HPT1 i"
            "nspection; score 3"
        ),
    },
    {
        "esn": "56180",
        "date": "2024-09-12",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_065",
        "label": (
            "ESN 56180 \u00b7 2024-09-12 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56007",
        "date": "2024-08-31",
        "kind": "HPT1 inspection",
        "score": None,
        "id": "event_066",
        "label": (
            "ESN 56007 \u00b7 2024-08-31 \u00b7 HPT1 i"
            "nspection; score unavailable"
        ),
    },
    {
        "esn": "56110",
        "date": "2024-08-25",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_067",
        "label": (
            "ESN 56110 \u00b7 2024-08-25 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56109",
        "date": "2024-08-25",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_068",
        "label": (
            "ESN 56109 \u00b7 2024-08-25 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56038",
        "date": "2024-08-19",
        "kind": "TRU leakage",
        "aircraft": "60026",
        "position": "RH",
        "osd": "61583",
        "id": "event_069",
        "label": "ESN 56038 \u00b7 2024-08-19 \u00b7 TRU leakage",
    },
    {
        "esn": "56037",
        "date": "2024-08-19",
        "kind": "TRU leakage",
        "aircraft": "60026",
        "position": "LH",
        "osd": "61583",
        "id": "event_070",
        "label": "ESN 56037 \u00b7 2024-08-19 \u00b7 TRU leakage",
    },
    {
        "esn": "56194",
        "date": "2024-07-19",
        "kind": "TRU leakage",
        "aircraft": "60086",
        "position": "RH",
        "osd": "61120",
        "id": "event_071",
        "label": "ESN 56194 \u00b7 2024-07-19 \u00b7 TRU leakage",
    },
    {
        "esn": "56099",
        "date": "2024-07-13",
        "kind": "TRU leakage",
        "aircraft": "60048",
        "position": "LH",
        "osd": "61031",
        "id": "event_072",
        "label": "ESN 56099 \u00b7 2024-07-13 \u00b7 TRU leakage",
    },
    {
        "esn": "56050",
        "date": "2024-06-26",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_073",
        "label": (
            "ESN 56050 \u00b7 2024-06-26 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56100",
        "date": "2024-05-28",
        "kind": "Other TRU event",
        "aircraft": "60048",
        "title": (
            "OSD60448 - A/C 60048 - RH ENG - FADEC FAU"
            "LT AND TRU FAIL MSG - LOW SPEED ABTO - TR"
            "U"
        ),
        "id": "event_074",
        "label": "ESN 56100 \u00b7 2024-05-28 \u00b7 Other TRU event",
    },
    {
        "esn": "56015",
        "date": "2024-02-23",
        "kind": "TRU leakage",
        "aircraft": "60007",
        "position": "LH",
        "osd": "59188; 59275",
        "id": "event_075",
        "label": "ESN 56015 \u00b7 2024-02-23 \u00b7 TRU leakage",
    },
    {
        "esn": "56045",
        "date": "2024-02-14",
        "kind": "TRU leakage",
        "aircraft": "60020",
        "position": "LH",
        "osd": "59067",
        "id": "event_076",
        "label": "ESN 56045 \u00b7 2024-02-14 \u00b7 TRU leakage",
    },
    {
        "esn": "56020",
        "date": "2024-01-10",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_077",
        "label": (
            "ESN 56020 \u00b7 2024-01-10 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56063",
        "date": "2023-12-16",
        "kind": "TRU leakage",
        "aircraft": "60031",
        "position": "LH",
        "osd": "58344; 58370",
        "id": "event_078",
        "label": "ESN 56063 \u00b7 2023-12-16 \u00b7 TRU leakage",
    },
    {
        "esn": "56013",
        "date": "2021-04-15",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_079",
        "label": (
            "ESN 56013 \u00b7 2021-04-15 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
    {
        "esn": "56012",
        "date": "2021-04-15",
        "kind": "HPT1 inspection",
        "score": 1,
        "id": "event_080",
        "label": (
            "ESN 56012 \u00b7 2021-04-15 \u00b7 HPT1 i"
            "nspection; score 1"
        ),
    },
]
GENIE_META = {
    "EngineId",
    "StartDatetime",
    "EndDatetime",
    "EngineSerialNumber",
    "ProvidedEngineSerialNumber",
    "ParentStartDatetime",
    "AircraftId",
    "AircraftIdentifier",
    "ProvidedAircraftIdentifier",
    "EnginePosition",
    "ProvidedEnginePosition",
    "OperatorId",
    "OperatorCode",
    "ProvidedOperatorCode",
    "LastGeneratedMessageId",
    "FirstGeneratedMessageId",
    "LastGeneratedDatetime",
    "FirstGeneratedDatetime",
    "LastReceivedMessageId",
    "FirstReceivedMessageId",
    "LastReceivedDatetime",
    "FirstReceivedDatetime",
    "Changed",
    "Created",
    "Migrated",
    "CalendarId",
    "AdditionalKey",
    "AIRFRAME_INPUT_ECHO",
}
GENIE_PREFIX = {"Take-off": "TO", "Cruise": "CZ"}
TRU_PATTERNS = (
    "OIP",
    "OIT",
    "VBHP",
    "VBLP",
    "HBV",
    "BLD",
    "EPR",
    "SAV",
    "SAIW",
    "W44",
    "W50",
    "P44",
    "P50",
    "PWXH",
)
GENIE_PRESETS = {
    "HPT1": ("gbm", 100, "all"),
    "TRU": ("rf", 100, "all"),
    "ANY": ("gbm", 100, "all"),
}
AUDIT_PRESETS = {
    "HPT1": ("gbm", 100, "all"),
    "TRU": ("rf_bal", 144, "tru"),
    "ANY": ("gbm", 100, "all"),
}


def genie_column_reason(column):
    normalized = str(column).upper()
    if normalized in {name.upper() for name in GENIE_META}:
        return "Metadata excluded by Genie"
    if re.search(r"_CONF_ASPA_[LU]\d?$|_OCC_[LU]$|_COUNTER_EWMA$|_PREV$", normalized):
        return "Confidence/occurrence/previous-value exclusion from Genie"
    if re.search("__NOM_|_NOM$|SMALLENGINEDUMMY", normalized):
        return "Nominal/reference/dummy exclusion from Genie"
    if re.search(
        "DATETIME|TIMESTAMP|MESSAGEID|SERIALNUMBER|IDENTIFIER",
        normalized,
    ):
        return (
            "Additional identity/time alias excluded; "
            "documented reconstruction safeguard"
        )
    return "Selected by Genie column rules"


def genie_feature_group(column):
    name = str(column).upper()
    if any(
        (
            token in name
            for token in ("DETECTED", "DRIFT", "LIMIT")
        ),
    ):
        return (
            "Existing detector indicator; generation r"
            "equires verification"
        )
    if any(
        (
            token in name
            for token in ("COUNT", "ACCUM", "PPV", "FLIGHT")
        ),
    ):
        return (
            "Accumulation/coverage candidate; audit fo"
            "r age or availability effects"
        )
    if any(
        (
            token in name
            for token in (
                "__DEL",
                "__MAR",
                "EWMA",
                "ASPA",
                "RATE_CHANGE",
                "MAR_AREA",
            )
        ),
    ):
        return "Engineered EPS performance/health indicator"
    if "PCA" in name:
        return (
            "PCA candidate; fleet applicability requir"
            "es verification"
        )
    return "Other numeric EPS measurement/indicator"


def prepare_eps_groups():
    snapshots = P15.get("snapshot_data") or P15["data"]
    groups, signals, rows = ({}, {}, [])
    for phase in PHASES:
        key = "".join(("{}".format(phase), " EPS"))
        frame = snapshots.get(key, pd.DataFrame())
        columns = [
            name
            for name in frame.attrs.get("signals", [])
            if name in frame and eps_allowed(name)
        ]
        if not columns:
            columns = [
                name
                for name in frame
                if (
                    not name.startswith("_")
                    and eps_allowed(name)
                    and numeric_schema(str(frame[name].dtype))
                )
            ]
        signals[phase] = columns
        groups[phase] = (
            {
                engine: data.sort_values("_time", kind="stable")
                for engine, data in frame.groupby("_engine", sort=False)
            }
            if not frame.empty
            else {}
        )
        schema = P15["schemas"].get(
            key,
            {name: str(frame[name].dtype) for name in columns},
        )
        for name, kind in schema.items():
            reason = genie_column_reason(name)
            rows.append(
                {
                    "Phase": phase,
                    "Column": name,
                    "Data type": kind,
                    "Selection": reason,
                    "Extracted for classifier": name in columns,
                    "Role": (
                        genie_feature_group(name)
                        if (
                            reason
                            == "Selected by Genie column rules"
                        )
                        else reason
                    ),
                },
            )
    P15["eps_groups"], P15["eps_columns"] = (groups, signals)
    P15["feature_inventory"] = pd.DataFrame(rows)
    return groups


def genie_snapshot_window(
    engine,
    cutoff,
    settings,
    category="Control",
    sample_id=None,
):
    engine, cutoff = (canonical_engine(engine), utc_stamp(cutoff))
    if "eps_groups" not in P15:
        prepare_eps_groups()
    result = {
        "sample_id": (
            sample_id
            or "".join(
                (
                    "{}".format(engine),
                    ":",
                    "{}".format(cutoff.isoformat()),
                    ":",
                    "{}".format(category),
                ),
            )
        ),
        "engine": engine,
        "category": category,
        "cutoff": cutoff,
        "label_known_at": (
            cutoff
            + pd.Timedelta(
                days=(
                    1
                    if category != "Control"
                    else settings["horizon_days"]
                ),
            )
        ),
        "control_verified": False,
        "flight_verified": bool(settings["takeoff_confirmed"]),
        "snapshot_identity": "EPS records; not verified flight cycles",
    }
    total = 0
    for phase, prefix in GENIE_PREFIX.items():
        columns = P15["eps_columns"].get(phase, [])
        frame = P15["eps_groups"].get(phase, {}).get(engine)
        sub = (
            frame.loc[frame["_time"].lt(cutoff)]
            if frame is not None
            else pd.DataFrame()
        )
        if not sub.empty and settings["strict_asof"]:
            sub = sub.loc[(
                sub["_available"].notna()
                & sub["_available"].lt(cutoff)
            )]
        if not sub.empty and settings.get("history_start"):
            sub = sub.loc[sub["_time"].ge(
                date_boundary(
                    settings["history_start"],
                    settings,
                ),
            )]
        if not sub.empty and settings.get("history_end"):
            sub = sub.loc[sub["_time"].lt(
                (
                    date_boundary(
                        settings["history_end"],
                        settings,
                    )
                    + pd.Timedelta(days=1)
                ),
            )]
        count = len(sub)
        result["".join(("quality|", "{}".format(phase), "|records"))] = count
        result["".join(("quality|", "{}".format(phase), "|gap_days"))] = (
            (cutoff - sub["_time"].max()).total_seconds() / 86400
            if count
            else np.nan
        )
        result["".join(
            ("quality|", "{}".format(phase), "|first_age_days"),
        )] = (
            (cutoff - sub["_time"].min()).total_seconds() / 86400
            if count
            else np.nan
        )
        result["".join(("quality|", "{}".format(phase), "|coverage"))] = 0.0
        if count < 5:
            continue
        recent = (
            sub.tail(10)[columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        means = recent.mean()
        valid = means.dropna()
        result.update(
            {
                "".join(
                    (
                        "{}".format(prefix),
                        ":",
                        "{}".format(name),
                        ":m",
                    ),
                ): float(value)
                for name, value in valid.items()
            },
        )
        result["".join(("quality|", "{}".format(phase), "|coverage"))
               ] = float(means.notna().mean()) if columns else 0.0
        total += len(valid)
        if count >= 25:
            prior = sub.iloc[-40:-10] if count >= 40 else sub.iloc[:-10]
            prior_means = (
                (
                    prior[columns].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                ).replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            ).mean()
            difference = means - prior_means
            result.update(
                {
                    "".join(
                        (
                            "{}".format(prefix),
                            ":",
                            "{}".format(name),
                            ":d",
                        ),
                    ): float(value)
                    for name, value in difference.dropna().items()
                },
            )
    result["observed_level_features"] = total
    result["evidence_status"] = (
        "EPS readings available"
        if total
        else (
            "No populated pre-event EPS features; benc"
            "hmark imputation must not be treated as a"
            "n assessment"
        )
    )
    return result


def genie_feature_columns(frame):
    return [
        name
        for name in frame
        if (
            isinstance(name, str)
            and name.startswith(("TO:", "CZ:"))
            and name.endswith((":m", ":d"))
        )
    ]


def build_genie_matrix(settings):
    prepare_eps_groups()
    event_engines = {canonical_engine(event["esn"]) for event in EVENTS}
    observed = set().union(
        *(set(groups) for groups in P15["eps_groups"].values()),
    )
    controls = sorted(observed - event_engines)
    by_engine = {}
    for event in EVENTS:
        (
            by_engine.setdefault(
                canonical_engine(event["esn"]),
                [],
            )
        ).append(
            event,
        )
    records = [
        genie_snapshot_window(
            event["esn"],
            date_boundary(event["date"], settings),
            settings,
            event_category(event),
            event["id"],
        )
        for events in by_engine.values()
        for event in events
    ]
    cutoff = utc_stamp(settings["control_cutoff"])
    records.extend(
        (
            genie_snapshot_window(
                engine,
                cutoff,
                settings,
                "Control",
                "".join(("genie-control:", "{}".format(engine))),
            )
            for engine in controls
        ),
    )
    frame = pd.DataFrame(records)
    columns = genie_feature_columns(frame)
    numeric = frame[columns].replace([np.inf, -np.inf], np.nan)
    coverage, deviation = (numeric.notna().mean(), numeric.std())
    keep = [
        name
        for name in columns
        if (
            coverage[name] >= 0.6
            and pd.notna(deviation[name])
            and (deviation[name] > 1e-12)
        )
    ]
    (
        P15["genie_matrix"],
        P15["genie_keep"],
        P15["control_engines"],
    ) = (frame, keep, controls)
    P15["feature_filter_report"] = pd.DataFrame(
        [
            {
                "Feature": name,
                "Coverage": coverage[name],
                "Standard deviation": deviation[name],
                "Kept in original global filter": name in keep,
            }
            for name in columns
        ],
    )
    return frame


def matching_distance(positive, negative):
    parts = []
    for phase in PHASES:
        if (
            positive.get(
                "".join(
                    ("quality|", "{}".format(phase), "|records"),
                ),
                0,
            )
            < 5
        ):
            continue
        if (
            negative.get(
                "".join(
                    ("quality|", "{}".format(phase), "|records"),
                ),
                0,
            )
            < 5
        ):
            return np.inf
        parts.append(
            abs(
                (
                    positive["".join(
                        (
                            "quality|",
                            "{}".format(phase),
                            "|coverage",
                        ),
                    )]
                    - negative["".join(
                        (
                            "quality|",
                            "{}".format(phase),
                            "|coverage",
                        ),
                    )]
                ),
            ),
        )
        for suffix in ("records", "gap_days", "first_age_days"):
            a, b = (
                positive.get(
                    "".join(
                        (
                            "quality|",
                            "{}".format(phase),
                            "|",
                            "{}".format(suffix),
                        ),
                    ),
                ),
                negative.get(
                    "".join(
                        (
                            "quality|",
                            "{}".format(phase),
                            "|",
                            "{}".format(suffix),
                        ),
                    ),
                ),
            )
            if pd.notna(a) and pd.notna(b):
                parts.append(
                    abs(
                        np.log1p(max(a, 0)) - np.log1p(max(b, 0)),
                    ),
                )
    return float(np.mean(parts)) if parts else np.inf


def build_matched_eps_cohort(settings, progress=print):
    rows, skipped, cache = ([], [], {})
    for index, event in enumerate(EVENTS):
        cutoff = (
            date_boundary(event["date"], settings)
            - pd.Timedelta(days=settings["lead_days"])
        )
        positive = genie_snapshot_window(
            event["esn"],
            cutoff,
            settings,
            event_category(event),
            event["id"],
        )
        if not positive["observed_level_features"]:
            skipped.append(
                {
                    "Sample": event["id"],
                    "Engine": event["esn"],
                    "Category": event_category(event),
                    "Reason": positive["evidence_status"],
                },
            )
            continue
        phases = tuple(
            (
                phase
                for phase in PHASES
                if (
                    positive["".join(
                        (
                            "quality|",
                            "{}".format(phase),
                            "|records",
                        ),
                    )]
                    >= 5
                )
            ),
        )
        candidates = []
        for engine in P15["control_engines"]:
            cache_key = (engine, cutoff)
            if cache_key not in cache:
                cache[cache_key] = genie_snapshot_window(engine, cutoff, settings)
            negative = dict(cache[cache_key])
            followup = cutoff + pd.Timedelta(days=settings["horizon_days"])
            ends = [
                P15["eps_groups"][phase][engine]["_time"].max()
                for phase in phases
                if engine in P15["eps_groups"][phase]
            ]
            if not ends or max(ends) < followup:
                continue
            distance = matching_distance(positive, negative)
            if (
                not np.isfinite(distance)
                or distance > settings["eps_match_distance"]
            ):
                continue
            for phase, prefix in GENIE_PREFIX.items():
                if phase not in phases:
                    negative = {
                        name: value
                        for name, value in negative.items()
                        if not name.startswith(prefix + ":")
                    }
                    for suffix in (
                        "records",
                        "gap_days",
                        "first_age_days",
                        "coverage",
                    ):
                        negative["".join(
                            (
                                "quality|",
                                "{}".format(phase),
                                "|",
                                "{}".format(suffix),
                            ),
                        )] = (
                            0
                            if suffix in ("records", "coverage")
                            else np.nan
                        )
            negative["sample_id"] = "".join(
                (
                    "control:",
                    "{}".format(engine),
                    ":",
                    "{}".format(cutoff.isoformat()),
                    ":",
                    "{}".format("-".join(phases)),
                ),
            )
            negative["operating_match_distance"] = distance
            candidates.append((distance, engine, negative))
        candidates.sort(
            key=lambda item: (
                item[0],
                (
                    hashlib.sha256(
                        (
                            "".join(
                                (
                                    "{}".format(settings["seed"]),
                                    ":",
                                    "{}".format(event["id"]),
                                    ":",
                                    "{}".format(item[1]),
                                ),
                            )
                        ).encode(),
                    )
                ).hexdigest(),
            ),
        )
        controls = [
            item[2]
            for item in candidates[:settings["matched_controls_per_event"]]
        ]
        if controls:
            rows.extend([positive] + controls)
        else:
            skipped.append(
                {
                    "Sample": event["id"],
                    "Engine": event["esn"],
                    "Category": "Control matching",
                    "Reason": (
                        "No control with comparable pr"
                        "e-event phase coverage/histor"
                        "y and follow-up at the same a"
                        "ssessment date; parameter rev"
                        "iew remains independent"
                    ),
                },
            )
        if (index + 1) % 10 == 0:
            progress(
                "".join(
                    (
                        "Matched EPS assessment dates: ",
                        "{}".format(index + 1),
                        "/",
                        "{}".format(len(EVENTS)),
                    ),
                ),
            )
    cohort = (
        (
            pd.DataFrame(rows).drop_duplicates("sample_id")
        ).reset_index(
            drop=True,
        )
        if rows
        else pd.DataFrame(
            columns=[
                "sample_id",
                "engine",
                "category",
                "cutoff",
                "label_known_at",
                "control_verified",
                "flight_verified",
            ],
        )
    )
    P15["cohort"], P15["cohort_skipped"] = (cohort, pd.DataFrame(skipped))
    return cohort


NOTEBOOK_VERSION = "6.2-GENIE-CODE"
OUTPUT_FAMILIES = {"P50", "P30", "T30", "VBHP", "VBLP", "TGT", "FF", "OIP", "OIT"}
PHASES = ("Take-off", "Cruise")
NAMESPACE = "ehm_fleetstore_prd1eun82719_internal.`global6500-pearl15`"
TABLE_NAMES = {
    "".join(("{}".format(phase), " ", "{}".format(kind.upper()))): "".join(
        (
            "{}".format(NAMESPACE),
            ".`",
            "{}".format(stem),
            "-aircraftengine-",
            "{}".format(kind),
            "`",
        ),
    )
    for phase, stem in [("Take-off", "takeoff"), ("Cruise", "cruise")]
    for kind in ("da", "eps")
}
DEFAULT_SETTINGS = {
    "max_flights": 300,
    "min_flights": 100,
    "lead_days": 0,
    "horizon_days": 30,
    "max_gap_days": 14,
    "flight_hours": 24,
    "controls_per_event": 4,
    "eps_per_phase": 10000,
    "selected_features": 35,
    "seed": 82719,
    "false_alarm_limit": 0.1,
    "detection_target": 0.8,
    "min_phase_fraction": 0.6,
    "max_missing_fraction": 0.5,
    "max_context_distance": 1.5,
    "takeoff_confirmed": False,
    "eps_causal_verified": False,
    "strict_asof": False,
    "baseline_confirmed": False,
    "event_timezone": "UTC",
    "max_rows": 1000000,
    "max_cells": 60000000,
    "inputs": [],
    "targets": [],
    "phases": list(PHASES),
    "history_start": None,
    "history_end": None,
    "eps_features": None,
    "allow_stale_history": True,
    "anchor_tolerance_minutes": 10,
    "cv_repeats": 2,
    "max_prediction_missing": 0.5,
    "feature_statistics": ["recent_mean"],
    "minimum_eps_cycles": 10,
    "raw_candidates": [],
    "train_end": None,
    "test_start": None,
}
P15 = {
    "schemas": {},
    "frames": {},
    "data": {},
    "diagnostics": [],
    "settings": None,
    "cohort": None,
    "validation": {},
    "raw_reviews": {},
    "replays": {},
    "models": {},
    "raw_errors": {},
    "history_warnings": {},
    "eps_reviews": {},
    "event_models": {},
    "healthy_intervals": pd.DataFrame(
        columns=["engine", "healthy_from", "healthy_to"],
    ),
}


def catalogue_session():
    session = globals().get("spark")
    if session is None:
        raise RuntimeError(
            (
                "Open this notebook on Databricks Pyth"
                "on compute with access to the Pearl-1"
                "5 catalogues."
            ),
        )
    return session


def canonical_engine(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"", "NULL", "NONE", "NAN", "NAT"}:
        return ""
    match = re.fullmatch("(?:ESN[\\s:_-]*)?([0-9]+)(?:\\.0+)?", text)
    return str(int(match.group(1))) if match else text


def utc_stamp(value, timezone="UTC"):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return pd.NaT
    return (
        (
            stamp.tz_localize(
                timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
        ).tz_convert(
            "UTC",
        )
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


def date_boundary(value, settings):
    return utc_stamp(
        pd.Timestamp(value).normalize(),
        settings["event_timezone"],
    )


def utc_series(values, timezone):
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    return (
        (
            parsed.dt.tz_localize(
                timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
        ).dt
    ).tz_convert(
        "UTC",
    )


def validate_settings(settings):
    for key in ("max_flights", "min_flights"):
        if not 100 <= int(settings[key]) <= 300:
            raise ValueError(
                (
                    "History must contain 100\u2013300"
                    " flight cycles."
                ),
            )
    if settings["min_flights"] > settings["max_flights"]:
        raise ValueError(
            "Minimum history cannot exceed maximum history.",
        )
    if (
        not (
            0
            <= int(settings["lead_days"])
            <= int(settings["horizon_days"])
            <= 90
        )
        or settings["horizon_days"] < 1
    ):
        raise ValueError(
            (
                "Lead time must be nonnegative and no "
                "longer than the prediction horizon (m"
                "aximum 90 days). The event day is alw"
                "ays excluded."
            ),
        )
    if not 0 < float(settings["false_alarm_limit"]) <= 0.1:
        raise ValueError("Choose a false-alarm budget up to 10%.")
    if not 0 < float(settings["min_phase_fraction"]) <= 1:
        raise ValueError(
            "Phase coverage must be between zero and one.",
        )
    if not 0 <= float(settings["max_missing_fraction"]) < 1:
        raise ValueError(
            "Missing-feature allowance must be below 100%.",
        )
    if (
        not settings["phases"]
        or not set(settings["phases"]).issubset(PHASES)
    ):
        raise ValueError("Select take-off, cruise, or both.")
    ZoneInfo(settings["event_timezone"])
    if (
        settings.get("history_start")
        and settings.get("history_end")
        and (
            pd.Timestamp(settings["history_start"])
            > pd.Timestamp(settings["history_end"])
        )
    ):
        raise ValueError("History start must precede its end.")
    if set(settings["inputs"]) & set(settings["targets"]):
        raise ValueError(
            (
                "A measured output cannot also be an o"
                "perating input."
            ),
        )
    families = [signal_family(name) for name in settings["inputs"]]
    if len(families) != len(set(families)):
        raise ValueError(
            (
                "Choose one operating input per family"
                "; duplicate ADC or RPM/% versions are"
                " redundant."
            ),
        )
    if any((family in OUTPUT_FAMILIES for family in families)):
        raise ValueError(
            "Keep monitored outputs out of operating inputs.",
        )
    if "EPR" in families:
        raise ValueError(
            (
                "EPR is excluded from operating inputs"
                " until its independence from measured"
                " outputs is established."
            ),
        )
    if len(settings["inputs"]) > 12 or len(settings["targets"]) > 8:
        raise ValueError(
            (
                "Select up to 12 operating inputs and "
                "eight measured outputs."
            ),
        )
    return settings


def numeric_schema(kind):
    return (
        (
            str(kind).lower()
            in {
                "double",
                "float",
                "int",
                "integer",
                "bigint",
                "long",
                "smallint",
                "short",
                "tinyint",
                "byte",
            }
        )
        or str(kind).lower().startswith("decimal")
    )


def field_named(schema, names):
    folded = {name.casefold(): name for name in schema}
    exact = next(
        (
            folded[name.casefold()]
            for name in names
            if name.casefold() in folded
        ),
        None,
    )
    if exact:
        return exact
    for wanted in names:
        normalized = re.sub("[^a-z0-9]", "", wanted.casefold())
        matches = [
            column
            for column in schema
            if (
                re.sub("[^a-z0-9]", "", column.casefold())
                == normalized
            )
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def timestamp_expression(columns, functions):
    columns = (
        [columns]
        if isinstance(columns, str)
        else list(columns or [])
    )
    parsed = [
        functions.expr(
            "".join(
                (
                    "try_cast(`",
                    "{}".format(column.replace("`", "``")),
                    "` as timestamp)",
                ),
            ),
        )
        for column in columns
    ]
    return (
        functions.coalesce(*parsed)
        if len(parsed) > 1
        else (
            parsed[0]
            if parsed
            else functions.lit(None).cast("timestamp")
        )
    )


def column_role_text(columns):
    return (
        columns[0]
        if len(columns) == 1
        else (
            "Row-wise fallback: " + " \u2192 ".join(columns)
            if columns
            else "Not populated"
        )
    )


def eps_allowed(column):
    return (
        genie_column_reason(column)
        == "Selected by Genie column rules"
    )


def eps_priority(column):
    name = str(column).upper()
    priority = sum(
        (
            weight
            for token, weight in [
                ("HPT1_FLT", 12),
                ("RATE_CHANGE", 10),
                ("EWMA", 8),
                ("__DEL_PC", 6),
                ("__MAR", 6),
                ("AREA", 5),
                ("ASPA", 3),
            ]
            if token in name
        ),
    )
    return (-priority, name)


def raw_measurement(column):
    name = str(column).upper()
    return (
        (
            signal_family(column)
            in (
                OUTPUT_FAMILIES
                | {
                    "P20",
                    "T20",
                    "ALT",
                    "NH",
                    "NL",
                    "RPM",
                    "MN",
                    "TRA",
                    "PACK",
                    "CAI",
                    "WAI",
                    "TAT",
                }
            )
        )
        and (not any(
            (
                token in name
                for token in (
                    "__NOM",
                    "_NOM",
                    "__DEL",
                    "__MAR",
                    "MAR_AREA",
                    "EWMA",
                    "ASPA",
                    "RATE_CHANGE",
                    "ACCUM",
                    "DETECTED",
                    "LIMIT",
                    "COUNT",
                    "PPV",
                    "PCA",
                )
            ),
        ))
    )


def discover_catalogues(table_names=TABLE_NAMES):
    session = catalogue_session()
    from pyspark.sql import functions as sf
    schemas, frames, rows, roles = ({}, {}, [], {})
    for key, path in table_names.items():
        phase, kind = key.rsplit(" ", 1)
        stem = "takeoff" if phase == "Take-off" else "cruise"
        expected = "".join(
            (
                "{}".format(stem),
                "-aircraftengine-",
                "{}".format(kind.lower()),
            ),
        )
        if (
            path.strip().split(".")[-1].strip("`").lower()
            != expected
        ):
            raise ValueError(
                "".join(
                    (
                        "{}".format(key),
                        ": use the ",
                        "{}".format(expected),
                        (
                            " table. Other flight phas"
                            "es are outside this analy"
                            "sis."
                        ),
                    ),
                ),
            )
        try:
            frame = session.table(path.strip())
            schema = {
                field.name: field.dataType.simpleString()
                for field in frame.schema.fields
            }
            schemas[key], frames[key] = (schema, frame)
            engine_fields = list(
                dict.fromkeys(
                    (
                        schema_name
                        for wanted in [
                            "EngineSerialNumber",
                            "ProvidedEngineSerialNumber",
                        ]
                        if (schema_name := field_named(schema, [wanted]))
                    ),
                ),
            )
            time_fields = list(
                dict.fromkeys(
                    (
                        schema_name
                        for wanted in [
                            "StartDatetime",
                            "SnapshotDatetime",
                            "SnapshotTimestamp",
                            "Timestamp",
                        ]
                        if (schema_name := field_named(schema, [wanted]))
                    ),
                ),
            )
            parent_fields = list(
                dict.fromkeys(
                    (
                        schema_name
                        for wanted in [
                            "ParentStartDatetime",
                            "ParentStartTimestamp",
                            "FlightStartDatetime",
                            "FlightStartTimestamp",
                        ]
                        if (schema_name := field_named(schema, [wanted]))
                    ),
                ),
            )
            probes = [sf.count(sf.lit(1)).alias("rows")]
            for column in dict.fromkeys(
                engine_fields + time_fields + parent_fields,
            ):
                probes.append(
                    (
                        sf.count(
                            sf.expr(
                                "".join(
                                    (
                                        "try_cast(`",
                                        "{}".format(
                                            column.replace("`", "``"),
                                        ),
                                        "` as ",
                                        "{}".format(
                                            (
                                                "string"
                                                if column in engine_fields
                                                else "timestamp"
                                            ),
                                        ),
                                        ")",
                                    ),
                                ),
                            ),
                        )
                    ).alias(
                        column,
                    ),
                )
            probes.extend(
                [
                    (
                        sf.count(
                            timestamp_expression(time_fields, sf),
                        )
                    ).alias(
                        "_sample_count",
                    ),
                    (
                        sf.count(
                            timestamp_expression(
                                parent_fields,
                                sf,
                            ),
                        )
                    ).alias(
                        "_parent_count",
                    ),
                ],
            )
            counts = frame.agg(*probes).collect()[0].asDict()
            engine = next(
                (
                    column
                    for column in engine_fields
                    if counts.get(column, 0)
                ),
                None,
            )
            populated_times = [
                column
                for column in time_fields
                if counts.get(column, 0)
            ]
            populated_parents = [
                column
                for column in parent_fields
                if counts.get(column, 0)
            ]
            time_field = populated_times[0] if populated_times else None
            parent = populated_parents[0] if populated_parents else None
            roles[key] = {
                "engine": engine,
                "engine_fields": engine_fields,
                "sample": time_field,
                "sample_fields": populated_times,
                "parent": parent,
                "parent_fields": populated_parents,
            }
            if parent:
                flight_field = column_role_text(populated_parents)
                flight_count = counts["_parent_count"]
                strategy = (
                    "Recorded flight starts; missing r"
                    "ows use reconciled take-off ancho"
                    "rs"
                )
            elif time_field and phase == "Take-off":
                flight_field = column_role_text(populated_times)
                flight_count = counts["_sample_count"]
                strategy = (
                    "Snapshot fallback; DA/EPS times r"
                    "econciled before counting flight "
                    "anchors"
                )
            elif time_field:
                flight_field = (
                    "Take-off anchor matched using "
                    + column_role_text(populated_times)
                )
                flight_count = "Calculated after take-off matching"
                strategy = (
                    "Cruise snapshots attach to preced"
                    "ing take-off; they do not add fli"
                    "ght cycles"
                )
            else:
                flight_field, flight_count, strategy = (
                    "Unavailable",
                    0,
                    (
                        "No usable snapshot or recorde"
                        "d flight-start timestamp"
                    ),
                )
            rows.append(
                {
                    "Table": key,
                    "Rows": counts["rows"],
                    "Engine field": engine or "Missing / empty",
                    "Snapshot field": column_role_text(populated_times),
                    "Populated snapshot times": counts["_sample_count"],
                    "Flight-start field": flight_field,
                    "Populated flight-start times": flight_count,
                    "Recorded flight-start field": column_role_text(populated_parents),
                    "Populated recorded flight-start times": counts["_parent_count"],
                    "Flight-start strategy": strategy,
                    "Eligible EPS signals": (
                        sum(
                            (
                                (
                                    eps_allowed(c)
                                    and (
                                        numeric_schema(t)
                                        or (
                                            t.lower()
                                            in {"string", "boolean"}
                                        )
                                    )
                                )
                                for c, t in schema.items()
                            ),
                        )
                        if kind == "EPS"
                        else None
                    ),
                    "Status": "Readable",
                },
            )
        except Exception as exc:
            rows.append(
                {
                    "Table": key,
                    "Status": "Unavailable",
                    "Reason": str(exc),
                },
            )
    P15["schemas"], P15["frames"] = (schemas, frames)
    P15["roles"] = roles
    P15["catalogue_report"] = pd.DataFrame(rows)
    schema_rows = []
    for phase in PHASES:
        da, eps = (
            schemas.get("".join(("{}".format(phase), " DA")), {}),
            schemas.get(
                "".join(("{}".format(phase), " EPS")),
                {},
            ),
        )
        missing = sorted(
            (
                column
                for column in da
                if (
                    column.casefold()
                    not in {name.casefold() for name in eps}
                )
            ),
        )
        schema_rows.append(
            {
                "Phase": phase,
                "DA columns": len(da),
                "EPS columns": len(eps),
                "DA columns absent from EPS": (
                    ", ".join(missing)
                    if missing
                    else (
                        "None"
                        if da and eps
                        else "Source unavailable"
                    )
                ),
                "Measurement source": (
                    (
                        "EPS raw measurements with DA "
                        "fallback; EPS-only columns re"
                        "tained"
                    )
                    if eps
                    else "DA only; EPS health channels unavailable"
                ),
            },
        )
    P15["schema_report"] = pd.DataFrame(schema_rows)
    if not any((key.endswith("EPS") for key in frames)):
        raise RuntimeError(
            (
                "Neither EPS table is accessible. Chec"
                "k the namespace and EPS table permiss"
                "ions. The table report contains the i"
                "ndividual errors."
            ),
        )
    return P15["catalogue_report"]


def default_operating_inputs(columns):
    choices = [
        ("P20_ADC1__PSI", "P20__PSI", "P20"),
        ("T20_ADC1__DEGC", "T20__DEGC", "T20__K"),
        ("ALT__FT", "ALT"),
        ("NH__RPM", "NH__PC", "RPM"),
        ("NL__RPM", "NL__PC"),
        ("MN", "MN_ADC1", "MACH"),
    ]
    folded = {name.upper(): name for name in columns}
    return [
        next(
            (
                folded[name]
                for name in alternatives
                if name in folded
            ),
        )
        for alternatives in choices
        if any((name in folded for name in alternatives))
    ]


def load_catalogues(settings, progress=print):
    validate_settings(settings)
    session = catalogue_session()
    if not P15["frames"]:
        discover_catalogues()
    from pyspark.sql import functions as sf
    try:
        source_timezone = (
            session.sql("SELECT current_timezone() AS timezone")
        ).collect()[0]["timezone"]
    except Exception as exc:
        raise RuntimeError(
            (
                "Cannot establish the catalogue timezo"
                "ne safely. SELECT current_timezone() "
                "must be available on this compute."
            ),
        ) from exc
    ZoneInfo(source_timezone)
    data, audit = ({}, [])
    total_cells = 0
    for key, source in P15["frames"].items():
        phase, kind = key.rsplit(" ", 1)
        if phase not in settings["phases"] and phase != "Take-off":
            continue
        schema = P15["schemas"][key]
        role = P15.get("roles", {}).get(key, {})
        engine = role.get("engine")
        sample = role.get("sample")
        parent = role.get("parent")
        if not engine or not sample:
            audit.append(
                {
                    "Table": key,
                    "Status": (
                        "Missing engine serial or snap"
                        "shot time; excluded"
                    ),
                },
            )
            continue
        timestamp_fields = [
            ("_time", role.get("sample_fields", [sample])),
            (
                "_parent",
                role.get(
                    "parent_fields",
                    [parent] if parent else [],
                ),
            ),
        ]
        for name, alternatives in [
            ("_created", ["Created", "FirstGeneratedDatetime"]),
            ("_changed", ["Changed", "LastGeneratedDatetime"]),
        ]:
            timestamp_fields.append(
                (
                    name,
                    list(
                        dict.fromkeys(
                            (
                                column
                                for wanted in alternatives
                                if (column := field_named(schema, [wanted]))
                            ),
                        ),
                    ),
                ),
            )
        engine_fields = role.get("engine_fields", [engine])
        projection = [
            (
                (
                    sf.col(
                        "".join(
                            (
                                "`",
                                "{}".format(
                                    column.replace("`", "``"),
                                ),
                                "`",
                            ),
                        ),
                    )
                ).cast(
                    "string",
                )
            ).alias(
                "".join(("_serial_", "{}".format(index))),
            )
            for index, column in enumerate(engine_fields)
        ]
        for alias, columns in timestamp_fields:
            parsed = timestamp_expression(columns, sf)
            projection.append(
                (
                    sf.date_format(
                        parsed,
                        "yyyy-MM-dd HH:mm:ss.SSSSSS",
                    )
                ).alias(
                    alias,
                ),
            )
        aircraft = field_named(
            schema,
            ["AircraftIdentifier", "ProvidedAircraftIdentifier"],
        )
        if aircraft:
            projection.append(
                (
                    (
                        sf.col(
                            "".join(
                                (
                                    "`",
                                    "{}".format(
                                        aircraft.replace("`", "``"),
                                    ),
                                    "`",
                                ),
                            ),
                        )
                    ).cast(
                        "string",
                    )
                ).alias(
                    "_aircraft",
                ),
            )
        if kind == "DA":
            signals = [
                column
                for column in dict.fromkeys(
                    (
                        settings["inputs"] + settings["targets"]
                        + settings.get("raw_candidates", [])
                    ),
                )
                if (
                    column in schema
                    and numeric_schema(schema[column])
                )
            ]
        else:
            candidates = sorted(
                [
                    c
                    for c, t in schema.items()
                    if (
                        (
                            numeric_schema(t)
                            or t.lower() in {"string", "boolean"}
                        )
                        and eps_allowed(c)
                    )
                ],
            )
            wanted = settings.get("eps_features")
            if wanted is not None:
                candidates = [
                    c
                    for c in candidates
                    if (
                        "".join(
                            (
                                "{}".format(phase),
                                "|",
                                "{}".format(c),
                            ),
                        )
                        in wanted
                    )
                ]
            signals = candidates[:int(settings["eps_per_phase"])]
            if not signals and phase in settings["phases"]:
                audit.append(
                    {
                        "Table": key,
                        "Status": (
                            "No permitted EPS features"
                            " selected; excluded"
                        ),
                    },
                )
                continue
        extras = [
            column
            for column in dict.fromkeys(
                (
                    settings["inputs"] + settings["targets"]
                    + settings.get("raw_candidates", [])
                ),
            )
            if (
                kind == "EPS"
                and column in schema
                and numeric_schema(schema[column])
                and (column not in signals)
            )
        ]
        for column in signals + extras:
            projection.append(
                (
                    sf.expr(
                        "".join(
                            (
                                "try_cast(`",
                                "{}".format(
                                    column.replace("`", "``"),
                                ),
                                "` as double)",
                            ),
                        ),
                    )
                ).alias(
                    column,
                ),
            )
        projected = source.select(*projection)
        if settings.get("history_start"):
            lower = (
                (
                    date_boundary(
                        settings["history_start"],
                        settings,
                    )
                ).tz_convert(
                    source_timezone,
                )
            ).strftime(
                "%Y-%m-%d %H:%M:%S.%f",
            )
            projected = projected.where(sf.col("_time") >= sf.lit(lower))
        if settings.get("history_end"):
            upper = (
                (
                    (
                        date_boundary(
                            settings["history_end"],
                            settings,
                        )
                        + pd.Timedelta(days=1)
                    )
                ).tz_convert(
                    source_timezone,
                )
            ).strftime(
                "%Y-%m-%d %H:%M:%S.%f",
            )
            projected = projected.where(sf.col("_time") < sf.lit(upper))
        row_count = projected.count()
        cells = row_count * len(projection)
        if (
            row_count > settings["max_rows"]
            or total_cells + cells > settings["max_cells"]
        ):
            raise ValueError(
                "".join(
                    (
                        "{}".format(key),
                        " would exceed the local-memory budget (",
                        "{:,}".format(row_count),
                        " rows, ",
                        "{:,}".format(cells),
                        (
                            " cells). Narrow the date "
                            "range or select fewer EPS"
                            " signals; rows will not b"
                            "e silently truncated."
                        ),
                    ),
                ),
            )
        progress(
            "".join(
                (
                    "Loading ",
                    "{}".format(key),
                    ": ",
                    "{:,}".format(row_count),
                    " rows, ",
                    "{}".format(len(signals)),
                    " signals",
                ),
            ),
        )
        local = projected.toPandas()
        if len(local) != row_count:
            raise RuntimeError(
                "".join(
                    (
                        "{}".format(key),
                        (
                            " changed during extractio"
                            "n. Reload to use a consis"
                            "tent dataset."
                        ),
                    ),
                ),
            )
        total_cells += cells
        serials = (
            local[[
                "".join(("_serial_", "{}".format(index)))
                for index in range(len(engine_fields))
            ]]
        ).apply(
            lambda column: column.map(canonical_engine),
        )
        serials = serials.replace("", np.nan)
        conflicting_serials = serials.nunique(axis=1, dropna=True).gt(1)
        resolved = serials.bfill(axis=1).iloc[:, 0].fillna("")
        local["_engine"] = resolved.mask(conflicting_serials, "")
        serial_conflicts = int(conflicting_serials.sum())
        local = local.drop(columns=list(serials.columns))
        for alias, _ in timestamp_fields:
            local[alias] = utc_series(local[alias], source_timezone)
        local["_available"] = local[["_created", "_changed"]].max(axis=1)
        before = len(local)
        local = (
            local.loc[local["_engine"].ne("") & local["_time"].notna()]
        ).copy()
        invalid_rows = before - len(local)
        duplicates = local.duplicated().sum()
        local = local.drop_duplicates().reset_index(drop=True)
        for column in signals:
            local[column] = pd.to_numeric(local[column], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if kind == "EPS":
            for column in extras:
                local[column] = (
                    pd.to_numeric(local[column], errors="coerce")
                ).replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
        if kind == "DA":
            for column in signals:
                valid = physical_validity(
                    local[column],
                    column,
                    {"physics_confirmed": False},
                )
                if (
                    signal_family(column) in {"TGT", "OIT"}
                    and (
                        measurement_unit(column)
                        in {"degC", "degF", "K"}
                    )
                ):
                    valid &= pd.Series(
                        to_kelvin(local[column], column) > 0,
                        index=local.index,
                    )
                local.loc[~valid, column] = np.nan
        local.attrs["signals"] = signals
        data[key] = local
        audit.append(
            {
                "Table": key,
                "Extracted rows": row_count,
                "Usable rows": len(local),
                "Invalid identity/time rows": invalid_rows,
                "Conflicting serial rows excluded": serial_conflicts,
                "Exact duplicates removed": int(duplicates),
                "Signals": len(signals),
                "Missing processing time": int(local["_available"].isna().sum()),
                "Post-snapshot processing rows": int((local["_available"] > local["_time"]).sum()),
                "Status": "Loaded",
            },
        )
    P15["data"], P15["load_report"] = (data, pd.DataFrame(audit))
    P15["source_timezone"] = source_timezone
    P15["settings"] = dict(settings)
    P15["snapshot_data"] = dict(data)
    try:
        assign_flight_cycles(settings)
    except ValueError as exc:
        if (
            "Neither take-off" not in str(exc)
            and "All take-off timestamps" not in str(exc)
        ):
            raise
        P15["inventory"] = pd.DataFrame(
            columns=["_engine", "_flight", "_cycle", "_identity"],
        )
        P15["flight_groups"], P15["engine_data"] = ({}, {key: {} for key in data})
        P15["data"] = {
            key: frame.iloc[:0].assign(
                _flight=pd.Series(dtype="datetime64[ns, UTC]"),
                _cycle=pd.Series(dtype=int),
                _identity=pd.Series(dtype=str),
            )
            for key, frame in data.items()
        }
        P15["cycle_report"] = pd.DataFrame(
            [
                {
                    "Status": (
                        "Flight-cycle parameter review"
                        " unavailable; independent EPS"
                        " classifier extraction retain"
                        "ed"
                    ),
                    "Reason": str(exc),
                },
            ],
        )
        P15["anchor_report"] = pd.DataFrame()
    return P15["load_report"]


def window_rows(key, engine, history, cutoff, settings):
    empty = P15["data"].get(key, pd.DataFrame()).iloc[:0].copy()
    rows = P15["engine_data"].get(key, {}).get(
        canonical_engine(engine),
        empty,
    )
    if rows.empty:
        return rows
    chosen = (
        rows.loc[(
            rows["_cycle"].isin(history["_cycle"])
            & rows["_time"].lt(cutoff)
        )]
    ).copy()
    if settings["strict_asof"] and key.endswith("EPS"):
        chosen = chosen.loc[(
            chosen["_available"].notna()
            & chosen["_available"].lt(cutoff)
        )]
    signals = list(
        dict.fromkeys(
            (
                P15["data"][key].attrs.get("signals", [])
                + [
                    column
                    for column in settings.get("raw_candidates", [])
                    if column in chosen
                ]
            ),
        ),
    )
    keys = ["_engine", "_time"]
    if signals and (not chosen.empty):
        counts = chosen.groupby(keys)[signals].nunique(dropna=False)
        conflicts = counts.gt(1).any(axis=1)
        if conflicts.any():
            rejected = set(conflicts.index[conflicts])
            chosen = chosen.loc[~pd.MultiIndex.from_frame(chosen[keys]).isin(rejected)]
        chosen = chosen.drop_duplicates(subset=keys)
    return chosen


def merge_measurement_sources(da, eps, settings):
    keys = ["_engine", "_time"]
    metadata = set(
        (
            keys
            + [
                "_parent",
                "_created",
                "_changed",
                "_available",
                "_aircraft",
                "_flight",
                "_cycle",
                "_identity",
            ]
        ),
    )
    requested = set(
        (
            settings["inputs"] + settings["targets"]
            + settings.get("raw_candidates", [])
        ),
    )
    raw_columns = sorted(
        (
            column
            for column in set(da.columns) | set(eps.columns)
            if column in requested and raw_measurement(column)
        ),
    )
    sources = []
    for source in (da, eps):
        selected = (
            source[[
                column
                for column in source
                if column in metadata or column in raw_columns
            ]]
        ).copy()
        if not selected.empty:
            for column in raw_columns:
                if column in selected:
                    selected[column] = (
                        pd.to_numeric(
                            selected[column],
                            errors="coerce",
                        )
                    ).replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    selected.loc[~physical_validity(
                        selected[column],
                        column,
                        {"physics_confirmed": False},
                    ), column] = np.nan
            if selected.duplicated(keys).any():
                raise ValueError(
                    (
                        "Measurement source contains a"
                        "mbiguous duplicate engine/sna"
                        "pshot rows."
                    ),
                )
            selected = selected.set_index(keys)
        sources.append(selected)
    left, right = sources
    if left.empty or right.empty:
        merged = (right if not right.empty else left).copy()
        merged = merged.reset_index() if not merged.empty else merged
        return (
            merged,
            {
                "EPS measurement snapshots": len(eps),
                "DA measurement snapshots": len(da),
                "DA-only snapshots": len(da) if eps.empty else 0,
                "Raw DA/EPS conflict cells excluded": 0,
                "Conflicting raw columns": "None",
            },
        )
    common = left.index.intersection(right.index)
    merged = right.combine_first(left)
    conflicts = {}
    for column in raw_columns:
        if column not in left or column not in right or common.empty:
            continue
        a, b = (left.loc[common, column], right.loc[common, column])
        conflict = (
            a.notna() & b.notna()
            & ~np.isclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-06,
                atol=1e-09,
                equal_nan=True,
            )
        )
        if conflict.any():
            merged.loc[common[conflict.to_numpy()], column] = np.nan
            conflicts[column] = int(conflict.sum())
    return (
        merged.reset_index(),
        {
            "EPS measurement snapshots": len(eps),
            "DA measurement snapshots": len(da),
            "DA-only snapshots": len(left.index.difference(right.index)),
            "Raw DA/EPS conflict cells excluded": sum(conflicts.values()),
            "Conflicting raw columns": ", ".join(conflicts) or "None",
        },
    )


def context_distance(left, right):
    distances = []
    for column in left:
        if not column.startswith("context|") or column not in right:
            continue
        scale_column = column.replace("context|", "context_scale|", 1)
        values = [
            left[column],
            right[column],
            left.get(scale_column, np.nan),
            right.get(scale_column, np.nan),
        ]
        if all(
            (
                pd.notna(value) and np.isfinite(value)
                for value in values
            ),
        ):
            scale = max(values[2], values[3], 1e-09)
            distances.append((values[0] - values[1]) / scale)
    return (
        float(np.sqrt(np.mean(np.square(distances))))
        if len(distances) >= 3
        else np.nan
    )


def event_category(event):
    return (
        "HPT1"
        if "HPT1" in event["kind"]
        else "HPT2" if "HPT2" in event["kind"] else "TRU"
    )


def healthy_confirmation(engine, start, end, intervals):
    if intervals.empty:
        return False
    matches = intervals.loc[intervals["engine"].eq(canonical_engine(engine))]
    return bool(
        (
            (
                (matches["healthy_from"] <= start)
                & (matches["healthy_to"] >= end)
            )
        ).any(),
    )


def parse_healthy_intervals(content, timezone="UTC"):
    frame = pd.read_csv(io.BytesIO(content), dtype={"engine": str})
    required = {"engine", "healthy_from", "healthy_to"}
    if not required.issubset(frame):
        raise ValueError(
            (
                "Healthy-history CSV needs engine, hea"
                "lthy_from, healthy_to columns. Includ"
                "e complete confirmed healthy periods."
            ),
        )
    frame = frame[list(required)].copy()
    frame["engine"] = frame["engine"].map(canonical_engine)
    for column in ("healthy_from", "healthy_to"):
        frame[column] = frame[column].map(lambda x: utc_stamp(x, timezone))
    if (
        frame["engine"].eq("").any()
        or frame[["healthy_from", "healthy_to"]].isna().any().any()
        or (frame["healthy_from"] > frame["healthy_to"]).any()
    ):
        raise ValueError(
            (
                "Healthy-history CSV contains empty en"
                "gines, invalid dates, or reversed int"
                "ervals."
            ),
        )
    return frame.drop_duplicates().reset_index(drop=True)


def assign_flight_cycles(settings):
    sources = []
    for key in ("Take-off DA", "Take-off EPS"):
        frame = P15["data"].get(key)
        if frame is not None and (not frame.empty):
            sources.append(
                frame[["_engine", "_time", "_parent"]].assign(
                    _source=key,
                ),
            )
    if not sources:
        raise ValueError(
            (
                "Neither take-off table has usable eng"
                "ine/timestamp rows. Cruise snapshots "
                "cannot establish flight cycles alone."
            ),
        )
    snapshots = pd.concat(sources, ignore_index=True).drop_duplicates()
    anchors, mappings, anchor_checks = ([], [], [])
    tolerance = pd.Timedelta(minutes=settings["anchor_tolerance_minutes"])
    for engine, group in snapshots.groupby("_engine", sort=True):
        rows = group.sort_values("_time").copy()
        rows["_anchor"] = rows["_parent"].where(
            (
                rows["_parent"].notna()
                & rows["_parent"].le(rows["_time"])
            ),
        )
        conflicts = (
            rows.loc[rows["_anchor"].notna()].groupby("_time")["_anchor"]
        ).nunique()
        conflict_times = set(conflicts.index[conflicts.gt(1)])
        rows = rows.loc[~rows["_time"].isin(conflict_times)].copy()
        known = (
            (
                rows.loc[rows["_anchor"].notna(), ["_time", "_anchor"]]
            ).drop_duplicates()
        ).sort_values(
            "_time",
        )
        recorded = set(known["_anchor"])
        missing = rows.loc[rows["_anchor"].isna()]
        if not missing.empty and (not known.empty):
            matched = pd.merge_asof(
                (
                    missing[["_time"]].assign(_row=missing.index)
                ).sort_values(
                    "_time",
                ),
                known.rename(columns={"_time": "_known_time"}),
                left_on="_time",
                right_on="_known_time",
                direction="nearest",
                tolerance=tolerance,
            )
            matched = matched.loc[matched["_anchor"].notna()].set_index(
                "_row",
            )
            rows.loc[matched.index, "_anchor"] = matched["_anchor"]
        fallback = (
            (
                rows.loc[rows["_anchor"].isna(), "_time"]
            ).drop_duplicates()
        ).sort_values()
        current = None
        replacements = {}
        for stamp in fallback:
            if current is None or stamp - current > tolerance:
                current = stamp
            replacements[stamp] = current
        if replacements:
            mask = rows["_anchor"].isna()
            rows.loc[mask, "_anchor"] = rows.loc[mask, "_time"].map(replacements)
        canonical = rows.groupby("_time")["_anchor"].nunique()
        ambiguous = set(canonical.index[canonical.gt(1)])
        rows = rows.loc[~rows["_time"].isin(ambiguous)]
        times = rows["_anchor"].dropna().drop_duplicates().sort_values()
        for position, stamp in enumerate(times):
            identity = (
                "Recorded parent flight starts"
                if stamp in recorded
                else (
                    "Reconciled take-off anchors verified"
                    if settings["takeoff_confirmed"]
                    else (
                        "Reconciled take-off snapshots"
                        "; unverified flight cycles"
                    )
                )
            )
            anchors.append(
                {
                    "_engine": engine,
                    "_flight": stamp,
                    "_cycle": position,
                    "_identity": identity,
                },
            )
        mappings.append(
            rows[["_engine", "_time", "_anchor"]].drop_duplicates(),
        )
        anchor_checks.append(
            {
                "Engine": engine,
                "DA snapshots": int(group["_source"].eq("Take-off DA").sum()),
                "EPS snapshots": int(group["_source"].eq("Take-off EPS").sum()),
                "Reconciled anchors": len(times),
                "Recorded parent anchors": sum((stamp in recorded for stamp in times)),
                "Conflicting timestamps excluded": len(conflict_times | ambiguous),
            },
        )
    if not anchors:
        raise ValueError(
            (
                "All take-off timestamps have conflict"
                "ing flight identity; no safe history "
                "can be reconstructed."
            ),
        )
    inventory = (
        pd.DataFrame(anchors).sort_values(["_engine", "_flight"])
    ).reset_index(
        drop=True,
    )
    inventory["_next"] = inventory.groupby("_engine")["_flight"].shift(-1)
    takeoff_lookup = pd.concat(mappings, ignore_index=True).rename(
        columns={"_anchor": "_flight"},
    )
    cycle_report = []
    for key, frame in list(P15["data"].items()):
        signals = frame.attrs.get("signals", [])
        chunks, unmapped = ([], 0)
        for engine, group in frame.groupby("_engine", sort=False):
            engine_inventory = inventory.loc[inventory["_engine"].eq(engine)]
            if engine_inventory.empty:
                unmapped += len(group)
                continue
            rows = (
                (
                    group.drop(
                        columns=["_flight", "_cycle", "_identity"],
                        errors="ignore",
                    )
                ).sort_values(
                    "_time",
                )
            ).copy()
            if key.startswith("Take-off"):
                lookup = (
                    takeoff_lookup.loc[takeoff_lookup["_engine"].eq(engine)]
                ).set_index(
                    "_time",
                )["_flight"]
                rows["_flight"] = rows["_time"].map(lookup)
            else:
                valid_parent = (
                    (
                        rows["_parent"].notna()
                        & rows["_parent"].isin(
                            engine_inventory["_flight"],
                        )
                    )
                    & rows["_parent"].le(rows["_time"])
                )
                rows["_flight"] = rows["_parent"].where(valid_parent)
                remaining = rows.loc[~valid_parent]
                if not remaining.empty:
                    matched = pd.merge_asof(
                        (
                            remaining[["_time"]].assign(
                                _row=remaining.index,
                            )
                        ).sort_values(
                            "_time",
                        ),
                        engine_inventory[["_flight", "_next"]],
                        left_on="_time",
                        right_on="_flight",
                        direction="backward",
                        tolerance=pd.Timedelta(
                            hours=settings["flight_hours"],
                        ),
                    )
                    usable = (
                        matched["_flight"].notna()
                        & (
                            matched["_next"].isna()
                            | matched["_time"].lt(matched["_next"])
                        )
                    )
                    matched = matched.loc[usable].set_index("_row")
                    rows.loc[matched.index, "_flight"] = matched["_flight"]
            elapsed = rows["_time"] - rows["_flight"]
            next_start = rows["_flight"].map(
                engine_inventory.set_index("_flight")["_next"],
            )
            valid = (
                (
                    (
                        rows["_flight"].notna()
                        & elapsed.ge(pd.Timedelta(0))
                    )
                    & elapsed.le(
                        pd.Timedelta(
                            hours=settings["flight_hours"],
                        ),
                    )
                )
                & (next_start.isna() | rows["_time"].lt(next_start))
            )
            unmapped += int((~valid).sum())
            rows = rows.loc[valid].copy()
            lookup = engine_inventory.set_index("_flight")
            rows["_cycle"] = rows["_flight"].map(lookup["_cycle"]).astype(int)
            rows["_identity"] = rows["_flight"].map(lookup["_identity"])
            chunks.append(rows)
        mapped = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else frame.iloc[:0].assign(
                _flight=pd.Series(dtype="datetime64[ns, UTC]"),
                _cycle=pd.Series(dtype=int),
                _identity=pd.Series(dtype=str),
            )
        )
        mapped.attrs["signals"] = signals
        P15["data"][key] = mapped
        cycle_report.append(
            {
                "Table": key,
                "Mapped snapshots": len(mapped),
                "Mapped engine-flight anchors": (
                    mapped[["_engine", "_cycle"]].drop_duplicates()
                ).shape[0],
                "Unmapped snapshots excluded": unmapped,
            },
        )
    P15["inventory"], P15["cycle_report"] = (inventory, pd.DataFrame(cycle_report))
    P15["anchor_report"] = pd.DataFrame(anchor_checks)
    P15["flight_groups"] = {
        engine: rows.copy()
        for engine, rows in inventory.groupby("_engine", sort=False)
    }
    P15["engine_data"] = {
        key: {
            engine: rows
            for engine, rows in frame.groupby("_engine", sort=False)
        }
        for key, frame in P15["data"].items()
    }
    return P15["cycle_report"]


def selected_history(engine, cutoff, settings):
    engine = canonical_engine(engine)
    inventory = P15["flight_groups"].get(engine)
    if inventory is None:
        raise ValueError(
            (
                "Neither take-off source has usable an"
                "chors for this engine after serial/ti"
                "mestamp checks."
            ),
        )
    chosen = inventory.loc[inventory["_flight"].lt(cutoff)]
    if settings.get("history_start"):
        chosen = chosen.loc[chosen["_flight"].ge(
            date_boundary(settings["history_start"], settings),
        )]
    if settings.get("history_end"):
        chosen = chosen.loc[chosen["_flight"].lt(
            (
                date_boundary(settings["history_end"], settings)
                + pd.Timedelta(days=1)
            ),
        )]
    chosen = chosen.tail(settings["max_flights"]).copy()
    if len(chosen) < settings["min_flights"]:
        raise ValueError(
            "".join(
                (
                    "Only ",
                    "{}".format(len(chosen)),
                    (
                        " prior reconciled flight anch"
                        "ors are available; at least "
                    ),
                    "{}".format(settings["min_flights"]),
                    " are required.",
                ),
            ),
        )
    gap = (cutoff - chosen["_flight"].iloc[-1]).total_seconds() / 86400
    if gap > settings["max_gap_days"]:
        message = "".join(
            (
                "Last available take-off is ",
                "{:.1f}".format(gap),
                (
                    " days before assessment. Historic"
                    "al evidence retained; recent cond"
                    "ition is unknown."
                ),
            ),
        )
        P15["history_warnings"][engine, cutoff.isoformat()] = message
        if not settings.get("allow_stale_history", True):
            raise ValueError(message)
    return chosen


def build_window(engine, cutoff, settings, metadata=None):
    cutoff = utc_stamp(cutoff)
    history = selected_history(engine, cutoff, settings)
    row = {
        "engine": canonical_engine(engine),
        "cutoff": cutoff,
        "history_cycles": len(history),
        "history_start": history["_flight"].iloc[0],
        "history_end": history["_flight"].iloc[-1],
        "flight_identity": "; ".join(history["_identity"].drop_duplicates()),
        "flight_verified": not history["_identity"].str.contains("unverified").any(),
        "gap_days": (
            (cutoff - history["_flight"].iloc[-1]).total_seconds()
            / 86400
        ),
    }
    row.update(metadata or {})
    informative = 0
    for phase in settings["phases"]:
        da = window_rows(
            "".join(("{}".format(phase), " DA")),
            engine,
            history,
            cutoff,
            settings,
        )
        eps = window_rows(
            "".join(("{}".format(phase), " EPS")),
            engine,
            history,
            cutoff,
            settings,
        )
        operating, _ = merge_measurement_sources(da, eps, settings)
        for column in settings["inputs"]:
            if column in operating:
                values = operating.groupby("_cycle")[column].median()
                row["".join(
                    (
                        "context|",
                        "{}".format(phase),
                        "|",
                        "{}".format(column),
                    ),
                )] = values.tail(10).median()
                row["".join(
                    (
                        "context_scale|",
                        "{}".format(phase),
                        "|",
                        "{}".format(column),
                    ),
                )] = (
                    max(
                        float(
                            (
                                values.quantile(0.75)
                                - values.quantile(0.25)
                            ),
                        ),
                        abs(float(values.median())) * 1e-06,
                        1e-09,
                    )
                    if values.notna().any()
                    else np.nan
                )
        source = P15["data"].get(
            "".join(("{}".format(phase), " EPS")),
            pd.DataFrame(),
        )
        signals = [
            name
            for name in source.attrs.get("signals", [])
            if name in eps
        ]
        cycles = (
            eps.groupby("_cycle")[signals].median().reindex(
                history["_cycle"],
            )
            if signals
            else pd.DataFrame(index=history["_cycle"])
        )
        populated = cycles.notna().any(axis=1)
        row["".join(("quality|", "{}".format(phase), "|coverage"))
            ] = float(populated.mean())
        row["".join(
            ("quality|", "{}".format(phase), "|eps_gap_days"),
        )] = (
            (cutoff - eps["_time"].max()).total_seconds() / 86400
            if not eps.empty
            else np.nan
        )
        recent = cycles.loc[populated].tail(10)
        means = (
            recent.mean()
            if (
                int(populated.sum())
                >= settings["minimum_eps_cycles"]
            )
            else pd.Series(np.nan, index=signals)
        )
        informative += int(means.notna().sum())
        for signal in source.attrs.get("signals", []):
            row["".join(
                (
                    "{}".format(phase),
                    "|",
                    "{}".format(signal),
                    "|recent_mean",
                ),
            )] = means.get(signal, np.nan)
    features = [
        name
        for name in row
        if name.count("|") == 2 and name.split("|", 1)[0] in PHASES
    ]
    row["missing_fraction"] = (
        float(pd.isna([row[name] for name in features]).mean())
        if features
        else 1.0
    )
    if informative < 3:
        raise ValueError(
            (
                "Fewer than three EPS channels have su"
                "fficient pre-event history. Raw measu"
                "rements are still reviewed separately"
                "."
            ),
        )
    return row


def build_cohort(settings, progress=print):
    event_engines = {canonical_engine(event["esn"]) for event in EVENTS}
    control_engines = sorted(set(P15["flight_groups"]) - event_engines)
    rows, skipped = ([], [])
    for position, event in enumerate(
        sorted(
            EVENTS,
            key=lambda event: (event["date"], event["id"]),
        ),
    ):
        event_time = date_boundary(event["date"], settings)
        cutoff = event_time - pd.Timedelta(days=settings["lead_days"])
        metadata = {
            "sample_id": event["id"],
            "event_id": event["id"],
            "category": event_category(event),
            "kind": event["kind"],
            "event_time": event_time,
            "label_known_at": event_time + pd.Timedelta(days=1),
            "control_verified": False,
        }
        try:
            positive = build_window(
                event["esn"],
                cutoff,
                settings,
                metadata,
            )
        except ValueError as exc:
            skipped.append(
                {
                    "Sample": event["id"],
                    "Engine": event["esn"],
                    "Category": metadata["category"],
                    "Reason": str(exc),
                },
            )
            continue
        candidates = sorted(
            control_engines,
            key=lambda engine: (
                hashlib.sha256(
                    (
                        "".join(
                            (
                                "{}".format(settings["seed"]),
                                "|",
                                "{}".format(cutoff.isoformat()),
                                "|",
                                "{}".format(engine),
                            ),
                        )
                    ).encode(),
                )
            ).hexdigest(),
        )
        negatives = []
        horizon_end = cutoff + pd.Timedelta(days=settings["horizon_days"])
        for engine in candidates:
            if len(negatives) >= settings["controls_per_event"]:
                break
            if (
                P15["flight_groups"][engine]["_flight"].max()
                < horizon_end
            ):
                continue
            meta = {
                "sample_id": "".join(
                    (
                        "control:",
                        "{}".format(engine),
                        ":",
                        "{}".format(cutoff.isoformat()),
                    ),
                ),
                "event_id": "",
                "category": "Control",
                "kind": "No listed incident",
                "event_time": pd.NaT,
                "label_known_at": horizon_end,
                "control_verified": False,
            }
            try:
                negative = build_window(engine, cutoff, settings, meta)
            except ValueError:
                continue
            if (
                abs(
                    (
                        negative["history_cycles"]
                        - positive["history_cycles"]
                    ),
                )
                > max(15, int(positive["history_cycles"] * 0.15))
            ):
                continue
            if any(
                (
                    (
                        abs(
                            (
                                negative["".join(
                                    (
                                        "quality|",
                                        "{}".format(phase),
                                        "|coverage",
                                    ),
                                )]
                                - positive["".join(
                                    (
                                        "quality|",
                                        "{}".format(phase),
                                        "|coverage",
                                    ),
                                )]
                            ),
                        )
                        > 0.2
                    )
                    for phase in settings["phases"]
                ),
            ):
                continue
            distance = context_distance(positive, negative)
            if (
                np.isfinite(distance)
                and distance > settings["max_context_distance"]
            ):
                continue
            negative["operating_match_distance"] = distance
            negative["control_verified"] = healthy_confirmation(
                engine,
                negative["history_start"],
                horizon_end,
                P15["healthy_intervals"],
            )
            negatives.append(negative)
        if negatives:
            rows.extend([positive] + negatives)
        else:
            skipped.append(
                {
                    "Sample": event["id"],
                    "Engine": event["esn"],
                    "Category": "Control matching",
                    "Reason": (
                        "No comparable control engine "
                        "is observed on the same asses"
                        "sment date with adequate foll"
                        "ow-up. Excluded from classifi"
                        "er validation; parameter revi"
                        "ew remains available."
                    ),
                },
            )
        if (position + 1) % 10 == 0:
            progress(
                "".join(
                    (
                        "Prepared ",
                        "{}".format(position + 1),
                        "/",
                        "{}".format(len(EVENTS)),
                        " incident dates",
                    ),
                ),
            )
    columns = [
        "sample_id",
        "engine",
        "category",
        "cutoff",
        "label_known_at",
    ]
    cohort = pd.DataFrame(rows)
    cohort = (
        (
            cohort.drop_duplicates("sample_id").sort_values(
                ["cutoff", "engine", "sample_id"],
            )
        ).reset_index(
            drop=True,
        )
        if not cohort.empty
        else pd.DataFrame(columns=columns)
    )
    P15["cohort"], P15["cohort_skipped"] = (cohort, pd.DataFrame(skipped))
    return cohort
