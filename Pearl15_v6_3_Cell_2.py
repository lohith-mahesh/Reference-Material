def runtime_model_notes(model, progress):
    constant = int(model.get("constant_feature_count", 0))
    invalid = int(model.get("undefined_feature_scores", 0))
    messages = model.get("fitting_warnings", [])
    if constant or invalid or messages:
        runtime_message(
            progress,
            "Feature ranking: {} constant columns; {} undefined scores; {} warnings recorded. "
            "Original Genie ranking is retained in reproduction mode.".format(
                constant, invalid, len(messages)
            ),
        )
        for message in messages:
            if "are constant" not in message:
                runtime_message(progress, "Training warning: {}".format(message))


class TrainingFeatureFilter(BaseEstimator, TransformerMixin):

    def __init__(self, minimum_present=0.65):
        self.minimum_present = minimum_present

    def fit(self, X, y=None):
        numeric = X.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        present = numeric.notna().mean()
        varying = numeric.nunique(dropna=True).gt(1)
        self.columns_ = (
            numeric.columns[(present >= self.minimum_present) & varying]
        ).tolist()
        if not self.columns_:
            raise ValueError(
                (
                    "No varying EPS features have adeq"
                    "uate coverage in the training eng"
                    "ines."
                ),
            )
        return self

    def transform(self, X):
        return (
            X.reindex(columns=self.columns_).apply(
                pd.to_numeric,
                errors="coerce",
            )
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )


def feature_columns(frame):
    return [
        column
        for column in frame
        if (
            column.count("|") == 2
            and column.split("|", 1)[0] in PHASES
            and eps_allowed(column.split("|")[1])
        )
    ]


def model_inputs(frame, kind, settings):
    selected = int(settings["selected_features"])
    classifier = (
        LogisticRegression(
            C=0.3,
            max_iter=2000,
            random_state=settings["seed"],
        )
        if kind == "Logistic"
        else HistGradientBoostingClassifier(
            max_iter=140,
            max_leaf_nodes=7,
            max_depth=3,
            min_samples_leaf=8,
            learning_rate=0.06,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=settings["seed"],
        )
    )
    return Pipeline(
        [
            ("coverage", TrainingFeatureFilter()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "selector",
                SelectKBest(score_func=f_classif, k=selected),
            ),
            ("scale", StandardScaler()),
            ("classifier", classifier),
        ],
    )


def engine_weights(frame, y):
    weights = (
        1.0
        / (
            frame.groupby("engine")["engine"].transform("size")
        ).to_numpy(
            dtype=float,
        )
    )
    for label in (0, 1):
        chosen = np.asarray(y) == label
        if chosen.any():
            weights[chosen] /= weights[chosen].sum()
    return weights * len(weights) / weights.sum()


def fit_classifier(train, kind, settings):
    columns = feature_columns(train)
    if not columns:
        raise ValueError(
            "No approved EPS features exist in this cohort.",
        )
    y = train["_label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            (
                "Both incident and control engines are"
                " required for model fitting."
            ),
        )
    estimator = model_inputs(train, kind, settings)
    probe = TrainingFeatureFilter().fit(train[columns])
    estimator.set_params(
        selector__k=min(settings["selected_features"], len(probe.columns_)),
    )
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sklearn.feature_selection",
        )
        estimator.fit(
            train[columns],
            y,
            classifier__sample_weight=engine_weights(train, y),
        )
    retained = estimator.named_steps["coverage"].columns_
    support = estimator.named_steps["selector"].get_support()
    selected = np.asarray(retained)[support].tolist()
    negatives = train.loc[train["_label"].eq(0)]
    return {
        "estimator": estimator,
        "columns": columns,
        "selected": selected,
        "control_center": negatives[selected].median(),
        "control_spread": (
            negatives[selected].quantile(0.75)
            - negatives[selected].quantile(0.25)
        ),
        "trained_engines": set(train["engine"]),
        "fit_known_until": train["label_known_at"].max(),
        "training_rows": len(train),
        "training_verified": bool(negatives["control_verified"].all()),
        "kind": kind,
    }


def predict_windows(model, frame, settings):
    if frame.empty:
        return pd.DataFrame(
            index=frame.index,
            columns=["score", "assessed", "reason"],
        )
    missing = frame.reindex(columns=model["selected"]).isna().mean(axis=1)
    assessed = missing.le(settings.get("max_prediction_missing", 0.5))
    score = pd.Series(np.nan, index=frame.index)
    if assessed.any():
        score.loc[assessed] = model["estimator"].predict_proba(
            frame.loc[assessed].reindex(columns=model["columns"]),
        )[:, 1]
    reason = pd.Series("", index=frame.index)
    reason.loc[~assessed] = "Too many selected EPS features are missing"
    return pd.DataFrame(
        {"score": score, "assessed": assessed, "reason": reason},
    )


def calibrated_threshold(controls, alpha):
    scores = (
        (
            (
                controls.dropna(subset=["score"]).groupby(
                    "engine",
                )["score"]
            ).max()
        ).sort_values()
    ).to_numpy()
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if n < 10 or rank > n:
        return (
            np.inf,
            n,
            (
                "Insufficient independent calibration "
                "controls for this false-alarm budget"
            ),
        )
    return (
        float(scores[rank - 1]),
        n,
        (
            "Empirical engine-level control threshold;"
            " exchangeability and prospective performa"
            "nce remain to be established"
        ),
    )


def binomial_interval(successes, total):
    if not total:
        return (np.nan, np.nan)
    return (
        (
            0.0
            if successes == 0
            else float(
                beta.ppf(0.025, successes, total - successes + 1),
            )
        ),
        (
            1.0
            if successes == total
            else float(
                beta.ppf(0.975, successes + 1, total - successes),
            )
        ),
    )


def clustered_detection_interval(positives, seed):
    if positives.empty:
        return (np.nan, np.nan)
    groups = [
        group["flag"].to_numpy(dtype=float)
        for _, group in positives.groupby("engine")
    ]
    complete_lower = binomial_interval(
        sum((bool(group.all()) for group in groups)),
        len(groups),
    )[0]
    any_upper = binomial_interval(
        sum((bool(group.any()) for group in groups)),
        len(groups),
    )[1]
    if len(groups) < 5:
        return (complete_lower, any_upper)
    rng = np.random.default_rng(seed)
    values = [
        (
            np.concatenate(
                [
                    groups[i]
                    for i in rng.integers(0, len(groups), len(groups))
                ],
            )
        ).mean()
        for _ in range(600)
    ]
    lower, upper = np.quantile(values, [0.025, 0.975]).tolist()
    return (min(lower, complete_lower), max(upper, any_upper))


def evaluation_metrics(test, threshold, settings):
    scored = test.loc[test["assessed"] & test["score"].notna()].copy()
    threshold_available = np.isfinite(threshold)
    scored["flag"] = (
        scored["score"].gt(threshold)
        if threshold_available
        else False
    )
    positives = scored.loc[scored["_label"].eq(1)]
    negatives = scored.loc[scored["_label"].eq(0)]
    control_flags = negatives.groupby("engine")["flag"].max()
    positive_engines = positives.groupby("engine")["flag"].max()
    detected = int(positives["flag"].sum())
    false_flags = int(control_flags.sum())
    roc_labels, roc_scores = validation_roc_arrays(scored)
    auc = (
        roc_auc_score(roc_labels, roc_scores)
        if len(np.unique(roc_labels)) == 2
        else np.nan
    )
    sensitivity = (
        detected / len(positives)
        if len(positives) and threshold_available
        else np.nan
    )
    false_rate = (
        false_flags / len(control_flags)
        if len(control_flags) and threshold_available
        else np.nan
    )
    sensitivity_ci = (
        clustered_detection_interval(positives, settings["seed"])
        if threshold_available
        else (np.nan, np.nan)
    )
    false_ci = (
        binomial_interval(false_flags, len(control_flags))
        if threshold_available
        else (np.nan, np.nan)
    )
    verified = bool(len(negatives) and negatives["control_verified"].all())
    point_met = bool(
        (
            np.isfinite(threshold)
            and np.isfinite(sensitivity)
            and np.isfinite(false_rate)
            and (sensitivity >= settings["detection_target"])
            and (false_rate <= settings["false_alarm_limit"])
        ),
    )
    supported = bool(
        (
            point_met
            and positives["engine"].nunique() >= 20
            and (len(control_flags) >= 60)
            and np.isfinite(sensitivity_ci[0])
            and (sensitivity_ci[0] >= settings["detection_target"])
            and (false_ci[1] <= settings["false_alarm_limit"])
        ),
    )
    return (
        {
            "Test incident windows": len(positives),
            "Detected incident windows": detected if threshold_available else np.nan,
            "Missed incident windows": (
                len(positives) - detected
                if threshold_available
                else np.nan
            ),
            "Incident engines": len(positive_engines),
            "Test control engines": len(control_flags),
            "Flagged control engines": false_flags if threshold_available else np.nan,
            "Detection rate": sensitivity,
            "Detection 95% lower": sensitivity_ci[0],
            "Detection 95% upper": sensitivity_ci[1],
            "Control flag rate": false_rate,
            "Control flag 95% lower": false_ci[0],
            "Control flag 95% upper": false_ci[1],
            "AUC": auc,
            "Unassessed test windows": (
                int((~test["assessed"]).sum())
                if threshold_available
                else len(test)
            ),
            "Controls verified healthy": verified,
            "Point target met": point_met,
            "Statistical evidence sufficient": supported,
        },
        scored,
    )


def validation_roc_arrays(scored):
    positive_scores = scored.loc[scored["_label"].eq(1), "score"].dropna().to_numpy()
    control_scores = (
        (
            (
                scored.loc[scored["_label"].eq(0)].dropna(
                    subset=["score"],
                )
            ).groupby(
                "engine",
            )["score"]
        ).max()
    ).to_numpy()
    return (
        np.r_[np.ones(len(positive_scores)), np.zeros(len(control_scores))],
        np.r_[positive_scores, control_scores],
    )


def confounding_audit(train, test, settings):
    columns = (
        [
            column
            for column in train
            if column.startswith("quality|")
        ]
        + ["history_cycles", "missing_fraction"]
    )

    def audit_data(frame):
        result = frame[columns].copy()
        result["calendar_year"] = frame["cutoff"].dt.year
        result["calendar_day"] = frame["cutoff"].dt.dayofyear
        result["history_span_days"] = (
            (
                (
                    (frame["history_end"] - frame["history_start"])
                ).dt
            ).total_seconds()
            / 86400
        )
        return result
    if train["_label"].nunique() < 2 or test["_label"].nunique() < 2:
        return {
            "AUC": np.nan,
            "Concern": "Insufficient classes for date/coverage audit",
        }
    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.3,
                    max_iter=1000,
                    random_state=settings["seed"],
                ),
            ),
        ],
    )
    estimator.fit(
        audit_data(train),
        train["_label"],
        classifier__sample_weight=engine_weights(train, train["_label"]),
    )
    scores = estimator.predict_proba(audit_data(test))[:, 1]
    auc = roc_auc_score(
        test["_label"],
        scores,
        sample_weight=engine_weights(test, test["_label"]),
    )
    return {
        "AUC": float(auc),
        "Concern": (
            (
                "Date/coverage alone separates labels;"
                " investigate before deployment"
            )
            if auc >= 0.7
            else (
                "No strong separation in this audit; r"
                "esidual confounding remains possible"
            )
        ),
    }


def eligible_head(cohort, head):
    selected = (
        cohort
        if head == "ANY"
        else cohort.loc[cohort["category"].isin([head, "Control"])]
    )
    selected = selected.copy().reset_index(drop=True)
    selected["_label"] = selected["category"].ne("Control").astype(int)
    return selected


def grouped_model_choice(train, settings):
    counts = train.groupby("_label")["engine"].nunique()
    if len(counts) < 2 or counts.min() < 6:
        return (
            "Logistic",
            pd.DataFrame(
                [
                    {
                        "Result": (
                            "Fixed regularised model; "
                            "too few engine groups for"
                            " model selection"
                        ),
                    },
                ],
            ),
        )
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=settings["seed"],
    )
    results = []
    for fold, (learning, held) in enumerate(
        splitter.split(train, train["_label"], train["engine"]),
    ):
        earlier, validation = (train.iloc[learning], train.iloc[held])
        for kind in ("Logistic", "Gradient boosting"):
            try:
                model = fit_classifier(earlier, kind, settings)
                prediction = predict_windows(model, validation, settings)
                valid = prediction["assessed"]
                labels = validation.loc[valid, "_label"]
                auc = (
                    roc_auc_score(
                        labels,
                        prediction.loc[valid, "score"],
                        sample_weight=engine_weights(
                            validation.loc[valid],
                            labels,
                        ),
                    )
                    if labels.nunique() == 2
                    else np.nan
                )
                results.append(
                    {"Fold": fold + 1, "Model": kind, "AUC": auc},
                )
            except ValueError as exc:
                results.append(
                    {
                        "Fold": fold + 1,
                        "Model": kind,
                        "AUC": np.nan,
                        "Reason": str(exc),
                    },
                )
    report = pd.DataFrame(results)
    valid = report.dropna(subset=["AUC"])
    averages = valid.groupby("Model")["AUC"].agg(["mean", "count"])
    averages = averages.loc[averages["count"].ge(2)]
    choice = (
        averages["mean"].idxmax()
        if not averages.empty
        else "Logistic"
    )
    if (
        choice == "Gradient boosting"
        and "Logistic" in averages.index
        and (
            averages.loc[choice, "mean"]
            < averages.loc["Logistic", "mean"] + 0.02
        )
    ):
        choice = "Logistic"
    return (choice, report)


def coverage_audit_prediction(train, test, settings):
    columns = (
        [name for name in train if name.startswith("quality|")]
        + ["history_cycles", "missing_fraction", "gap_days"]
    )

    def audit_frame(frame):
        result = frame.reindex(columns=columns).copy()
        result["calendar_year"] = frame["cutoff"].dt.year
        result["calendar_day"] = frame["cutoff"].dt.dayofyear
        result["history_span_days"] = (
            (
                (
                    (frame["history_end"] - frame["history_start"])
                ).dt
            ).total_seconds()
            / 86400
        )
        return result.replace([np.inf, -np.inf], np.nan)
    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=80,
                    max_depth=2,
                    max_leaf_nodes=4,
                    min_samples_leaf=8,
                    l2_regularization=5,
                    early_stopping=False,
                    random_state=settings["seed"],
                ),
            ),
        ],
    )
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("ignore", category=UserWarning)
        estimator.fit(
            audit_frame(train),
            train["_label"],
            classifier__sample_weight=engine_weights(train, train["_label"]),
        )
        return estimator.predict_proba(audit_frame(test))[:, 1]


def cross_validation_metrics(predictions, settings):
    scored = (
        predictions.loc[predictions["assessed"] & predictions["score"].notna()]
    ).copy()
    evaluated = scored.loc[scored["alert_assessed"]].copy()
    positive = evaluated.loc[evaluated["_label"].eq(1)]
    negative = evaluated.loc[evaluated["_label"].eq(0)]
    controls = negative.groupby("engine")["flag"].max()
    detection = (
        float(positive["flag"].mean())
        if not positive.empty
        else np.nan
    )
    control_rate = float(controls.mean()) if len(controls) else np.nan
    roc_labels, roc_scores = validation_roc_arrays(scored)
    detection_interval = (
        clustered_detection_interval(positive, settings["seed"])
        if not positive.empty
        else (np.nan, np.nan)
    )
    control_interval = binomial_interval(int(controls.sum()), len(controls))
    audit = scored.dropna(subset=["audit_score"])
    audit_auc = (
        roc_auc_score(
            audit["_label"],
            audit["audit_score"],
            sample_weight=engine_weights(audit, audit["_label"]),
        )
        if audit["_label"].nunique() == 2
        else np.nan
    )
    return {
        "Detection rate": detection,
        "Control flag rate": control_rate,
        "AUC": (
            roc_auc_score(roc_labels, roc_scores)
            if len(np.unique(roc_labels)) == 2
            else np.nan
        ),
        "Test incident windows": len(positive),
        "Detected incident windows": int(positive["flag"].sum()),
        "Incident engines": positive["engine"].nunique(),
        "Test control engines": len(controls),
        "Flagged control engines": int(controls.sum()),
        "Unassessed test windows": int((~predictions["alert_assessed"]).sum()),
        "Assessment coverage": (
            float(predictions["alert_assessed"].mean())
            if not predictions.empty
            else np.nan
        ),
        "Detection 95% lower": detection_interval[0],
        "Detection 95% upper": detection_interval[1],
        "Control flag 95% lower": control_interval[0],
        "Control flag 95% upper": control_interval[1],
        "Date/coverage-only AUC": audit_auc,
        "Controls verified healthy": bool(
            (
                not negative.empty
                and negative["control_verified"].all()
            ),
        ),
        "Point target met": bool(
            (
                pd.notna(detection)
                and pd.notna(control_rate)
                and (detection >= settings["detection_target"])
                and (control_rate <= settings["false_alarm_limit"])
            ),
        ),
    }


def validate_grouped_head(cohort, head, settings, progress=print):
    chosen = eligible_head(cohort, head)
    engines = chosen.groupby("engine")["_label"].max().reset_index()
    counts = engines["_label"].value_counts().reindex(
        [0, 1],
        fill_value=0,
    )
    result = {
        "head": head,
        "status": "Classifier not established",
        "metrics": {},
        "test": pd.DataFrame(),
        "all_predictions": pd.DataFrame(),
        "repeat_metrics": pd.DataFrame(),
        "folds": pd.DataFrame(),
        "warnings": [],
        "models": {},
    }
    if counts[1] < 6 or counts[0] < 12:
        result["warnings"] = [
            "".join(
                (
                    "{}".format(counts[1]),
                    " independent incident engines and ",
                    "{}".format(counts[0]),
                    (
                        " control engines are eligible"
                        ". At least 6 and 12 respectiv"
                        "ely are required for retrospe"
                        "ctive grouped validation. EPS"
                        " and raw parameter reviews ru"
                        "n independently."
                    ),
                ),
            ),
        ]
        return result
    n_splits = min(5, int(counts.min()))
    all_predictions, all_folds, repeat_metrics = ([], [], [])
    P15["event_models"][head] = {}
    for repeat in range(settings["cv_repeats"]):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=settings["seed"] + repeat,
        )
        predictions = []
        for fold, (learning, held) in enumerate(
            splitter.split(
                engines,
                engines["_label"],
                engines["engine"],
            ),
        ):
            pool_engines = set(engines.iloc[learning]["engine"])
            held_engines = set(engines.iloc[held]["engine"])
            pool = chosen.loc[chosen["engine"].isin(pool_engines)]
            test = chosen.loc[chosen["engine"].isin(held_engines)].copy()
            control_engines = sorted(
                pool.loc[pool["_label"].eq(0), "engine"].unique(),
                key=lambda engine: (
                    hashlib.sha256(
                        (
                            "".join(
                                (
                                    "{}".format(settings["seed"]),
                                    ":",
                                    "{}".format(repeat),
                                    ":",
                                    "{}".format(fold),
                                    ":",
                                    "{}".format(engine),
                                ),
                            )
                        ).encode(),
                    )
                ).hexdigest(),
            )
            minimum_calibration = max(
                10,
                (
                    int(
                        np.ceil(
                            1 / settings["false_alarm_limit"],
                        ),
                    )
                    - 1
                ),
            )
            calibration_count = min(
                max(
                    minimum_calibration,
                    int(np.ceil(len(control_engines) * 0.25)),
                ),
                max(0, len(control_engines) - 5),
            )
            calibration_engines = set(control_engines[:calibration_count])
            train = pool.loc[~pool["engine"].isin(calibration_engines)]
            calibration = pool.loc[pool["engine"].isin(calibration_engines)]
            if (
                set(train["engine"]) & held_engines
                or calibration_engines & held_engines
                or set(train["engine"]) & calibration_engines
            ):
                raise AssertionError(
                    (
                        "Training, calibration and hel"
                        "d-out engine identities overl"
                        "ap."
                    ),
                )
            output = test.copy()
            (
                output["score"],
                output["audit_score"],
                output["alert_threshold"],
            ) = (np.nan, np.nan, np.nan)
            output["assessed"], output["alert_assessed"] = (False, False)
            output["flag"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
            output["reason"] = "Model not fitted"
            try:
                kind, selection = grouped_model_choice(train, settings)
                model = fit_classifier(train, kind, settings)
                cal_prediction = predict_windows(model, calibration, settings)
                cal = pd.concat([calibration, cal_prediction], axis=1)
                threshold, independent_controls, threshold_note = calibrated_threshold(
                    cal.loc[cal["assessed"]],
                    settings["false_alarm_limit"],
                )
                model.update(
                    {
                        "threshold": threshold,
                        "calibration_engines": calibration_engines,
                        "head": head,
                        "evaluation": (
                            "Retrospective engine-held"
                            "-out cross-validation; no"
                            "t an earlier-only deploym"
                            "ent replay"
                        ),
                    },
                )
                prediction = predict_windows(model, test, settings)
                output[["score", "assessed", "reason"]
                       ] = prediction[["score", "assessed", "reason"]]
                output["audit_score"] = coverage_audit_prediction(train, test, settings)
                output["alert_threshold"] = threshold
                output["alert_assessed"] = output["assessed"] & np.isfinite(threshold)
                output.loc[output["alert_assessed"], "flag"] = (
                    output.loc[output["alert_assessed"], "score"].gt(
                        threshold,
                    )
                )
                output.loc[output["assessed"] & ~
                           output["alert_assessed"], "reason"] = threshold_note
                model_key = (repeat, fold)
                result["models"][model_key] = model
                if repeat == 0:
                    for event_id in test.loc[test["_label"].eq(1), "sample_id"]:
                        P15["event_models"][head][event_id] = model
                all_folds.append(
                    {
                        "Repeat": repeat + 1,
                        "Fold": fold + 1,
                        "Training engines": train["engine"].nunique(),
                        "Calibration control engines": independent_controls,
                        "Held-out engines": len(held_engines),
                        "Model": kind,
                        "Threshold": threshold,
                        "Reason": threshold_note,
                    },
                )
            except ValueError as exc:
                output["reason"] = str(exc)
                all_folds.append(
                    {
                        "Repeat": repeat + 1,
                        "Fold": fold + 1,
                        "Training engines": train["engine"].nunique(),
                        "Calibration control engines": calibration["engine"].nunique(),
                        "Held-out engines": len(held_engines),
                        "Model": "Unavailable",
                        "Threshold": np.nan,
                        "Reason": str(exc),
                    },
                )
            output["repeat"], output["fold"] = (repeat + 1, fold + 1)
            predictions.append(output)
        combined = pd.concat(predictions, ignore_index=True)
        if (
            combined["sample_id"].duplicated().any()
            or (
                set(combined["sample_id"])
                != set(chosen["sample_id"])
            )
        ):
            raise AssertionError(
                (
                    "Every eligible window must appear"
                    " once per cross-validation repeat"
                    "."
                ),
            )
        all_predictions.append(combined)
        repeat_metrics.append(
            {
                "Repeat": repeat + 1,
                **cross_validation_metrics(combined, settings),
            },
        )
        progress(
            "".join(
                (
                    "{}".format(head),
                    ": grouped validation repeat ",
                    "{}".format(repeat + 1),
                    "/",
                    "{}".format(settings["cv_repeats"]),
                    " complete",
                ),
            ),
        )
    primary = all_predictions[0]
    metrics = cross_validation_metrics(primary, settings)
    warnings_list = [
        (
            "Retrospective evaluation uses other engin"
            "es' historical outcomes, including later "
            "outcomes. It does not prove a model was a"
            "vailable before each event."
        ),
    ]
    if not metrics["Controls verified healthy"]:
        warnings_list.append(
            (
                "Control engines have no listed incide"
                "nts; their health is unverified. Cont"
                "rol flags are not a confirmed healthy"
                "-engine false-positive rate."
            ),
        )
    if (
        "operating_match_distance" not in chosen
        or (
            (
                chosen.loc[chosen["_label"].eq(0), "operating_match_distance"]
            ).isna()
        ).any()
    ):
        warnings_list.append(
            (
                "Comparable operating conditions could"
                " not be verified for some controls; E"
                "PS correction meanings require engine"
                "ering review."
            ),
        )
    if metrics["Date/coverage-only AUC"] >= 0.7:
        warnings_list.append(
            (
                "Date/history/coverage alone distingui"
                "shes incident labels; investigate con"
                "founding before relying on the classi"
                "fier."
            ),
        )
    if metrics["Unassessed test windows"]:
        warnings_list.append(
            "".join(
                (
                    "{}".format(
                        metrics["Unassessed test windows"],
                    ),
                    (
                        " windows lack sufficient sele"
                        "cted features or an independe"
                        "ntly calibrated alert thresho"
                        "ld."
                    ),
                ),
            ),
        )
    if not settings["eps_causal_verified"]:
        warnings_list.append(
            (
                "EPS meanings and earlier-only feature"
                " generation require engineering confi"
                "rmation."
            ),
        )
    if not bool(chosen["flight_verified"].all()):
        warnings_list.append(
            (
                "Some histories use inferred take-off "
                "anchors; genuine flight-cycle identit"
                "y requires confirmation."
            ),
        )
    age_columns = [name for name in chosen if name.endswith("eps_gap_days")]
    if (
        age_columns
        and (
            chosen[age_columns].gt(settings["max_gap_days"]).any()
        ).any()
    ):
        warnings_list.append(
            "".join(
                (
                    "Some EPS histories end more than ",
                    "{}".format(settings["max_gap_days"]),
                    (
                        " days before assessment. Thei"
                        "r historical evidence is reta"
                        "ined; recent engine condition"
                        " is unknown."
                    ),
                ),
            ),
        )
    result.update(
        {
            "status": (
                "Retrospective classifier assessed"
                if primary["alert_assessed"].any()
                else (
                    (
                        "Scores available; alert thres"
                        "hold not established"
                    )
                    if primary["assessed"].any()
                    else "Classifier not established"
                )
            ),
            "metrics": metrics,
            "test": primary,
            "all_predictions": pd.concat(all_predictions, ignore_index=True),
            "repeat_metrics": pd.DataFrame(repeat_metrics),
            "folds": pd.DataFrame(all_folds),
            "warnings": warnings_list,
        },
    )
    return result


def run_validation(settings, progress=print):
    cohort = (
        P15["cohort"]
        if P15["cohort"] is not None
        else build_cohort(settings, progress)
    )
    results = {
        head: validate_grouped_head(cohort, head, settings, progress)
        for head in ("HPT1", "HPT2", "TRU", "ANY")
    }
    P15["validation"] = results
    P15["manifest"] = {
        "version": NOTEBOOK_VERSION,
        "settings": settings,
        "tables": P15.get("table_names", TABLE_NAMES),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "validation": (
            "Repeated engine-grouped outer cross-valid"
            "ation; model selection and preprocessing "
            "inside training; separate calibration con"
            "trols; matched assessment dates; primary "
            "metrics from prespecified first repeat"
        ),
        "feature_matrix": {
            "windows": len(cohort),
            "engines": cohort["engine"].nunique(),
            "EPS_features": len(feature_columns(cohort)),
        },
        "limitations": (
            "Retrospective evidence; processing-time/h"
            "ealthy-control/flight-identity verificati"
            "on may remain incomplete. Scores and para"
            "meter deviation percentages are not failu"
            "re probabilities."
        ),
    }
    return results


def genie_estimator(kind):
    if kind == "gbm":
        return GradientBoostingClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.08,
            min_samples_leaf=5,
            subsample=0.85,
            random_state=42,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=4,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        )
    if kind == "rf_bal":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )
    if kind == "lr":
        return LogisticRegression(
            C=0.3,
            max_iter=2000,
            class_weight="balanced",
        )
    raise ValueError(
        "".join(("Unsupported classifier: ", "{}".format(kind))),
    )


def genie_model_fit(train, config, columns=None, training_only=True):
    kind, topk, subset = config
    columns = list(
        (
            columns
            if columns is not None
            else genie_feature_columns(train)
        ),
    )
    if subset == "tru":
        columns = [
            name
            for name in columns
            if any(
                (
                    pattern in name.upper()
                    for pattern in TRU_PATTERNS
                ),
            )
        ]
    numeric = (
        train.reindex(columns=columns).apply(
            pd.to_numeric,
            errors="coerce",
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if training_only:
        columns = [
            name
            for name in columns
            if (
                numeric[name].notna().mean() >= 0.6
                and pd.notna(numeric[name].std())
                and (numeric[name].std() > 1e-12)
            )
        ]
        numeric = numeric[columns]
    if not columns:
        raise ValueError(
            (
                "No varying EPS level/change features "
                "have sufficient training coverage"
            ),
        )
    median = numeric.median()
    columns = [name for name in columns if pd.notna(median[name])]
    if not columns:
        raise ValueError(
            "All candidate EPS features are missing in training",
        )
    numeric, median = (numeric[columns], median[columns])
    values = numeric.fillna(median).to_numpy(dtype=float)
    constant_mask = np.ptp(values, axis=0) == 0
    if training_only and constant_mask.any():
        retained = ~constant_mask
        if not retained.any():
            raise ValueError("No non-constant training features remain after imputation")
        columns = [name for name, keep in zip(columns, retained) if keep]
        numeric, median = numeric[columns], median[columns]
        values = values[:, retained]
        constant_mask = np.zeros(len(columns), dtype=bool)
    labels = train["_label"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Both event and control observations are required",
        )
    with warnings.catch_warnings(record=True) as fitting_warnings, threadpool_limits(limits=2):
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        importance, _ = f_classif(values, labels)
        undefined_scores = int(np.isnan(importance).sum())
        importance = np.nan_to_num(importance, nan=0.0)
        index = np.argsort(importance)[::-1][:min(topk, len(columns))]
        selected = [columns[i] for i in index]
        scaler = StandardScaler().fit(values[:, index])
        estimator = genie_estimator(kind).fit(
            scaler.transform(values[:, index]),
            labels,
        )
    return {
        "estimator": estimator,
        "scaler": scaler,
        "median": median[selected],
        "selected": selected,
        "columns": selected,
        "trained_engines": set(train["engine"]),
        "control_center": numeric.loc[train["_label"].eq(0), selected].median(),
        "kind": kind,
        "config": config,
        "training_only_preprocessing": training_only,
        "constant_feature_count": int(constant_mask.sum()),
        "undefined_feature_scores": undefined_scores,
        "fitting_warnings": [
            "{}: {}".format(item.category.__name__, str(item.message)[:600])
            for item in fitting_warnings
        ],
    }


def genie_predict(model, frame, settings, assess=True):
    numeric = (
        frame.reindex(columns=model["selected"]).apply(
            pd.to_numeric,
            errors="coerce",
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    valid = (
        (
            numeric.notna().mean(axis=1).ge(
                1 - settings["max_prediction_missing"],
            )
            & (
                frame.get(
                    "observed_level_features",
                    pd.Series(1, index=frame.index),
                )
            ).gt(
                0,
            )
        )
        if assess
        else pd.Series(True, index=frame.index)
    )
    result = pd.DataFrame(
        {"score": np.nan, "assessed": valid, "reason": ""},
        index=frame.index,
    )
    if valid.any():
        with threadpool_limits(limits=2):
            values = numeric.loc[valid].fillna(model["median"]).to_numpy(
                dtype=float,
            )
            result.loc[valid, "score"] = model["estimator"].predict_proba(
                model["scaler"].transform(values),
            )[:, 1]
    result.loc[~valid, "reason"] = (
        "No EPS evidence or more than half of selected"
        " features missing; not assessed by imputation"
        " alone"
    )
    return result


def genie_roc_point(labels, scores, limit):
    labels, scores = (np.asarray(labels), np.asarray(scores, dtype=float))
    valid = np.isfinite(scores)
    labels, scores = (labels[valid], scores[valid])
    if len(np.unique(labels)) < 2:
        return (np.nan, np.nan, np.nan)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    allowed = fpr <= limit
    index = int(np.argmax(tpr * allowed))
    return (
        float(tpr[index]),
        float(fpr[index]),
        float(thresholds[index]),
    )


def roc_profile(predictions, stage, head, config):
    scores = predictions.groupby("sample_id", sort=False)["score"].mean()
    meta = predictions.drop_duplicates("sample_id").set_index(
        "sample_id",
    )
    valid = scores.notna()
    labels = meta.loc[scores.index[valid], "_label"].to_numpy(dtype=int)
    values = scores.loc[valid].to_numpy(dtype=float)
    row = {
        "Experiment": stage,
        "Issue": head,
        "Model": config[0],
        "K": config[1],
        "Subset": config[2],
        "Assessed incidents": int((labels == 1).sum()),
        "Assessed control windows": int((labels == 0).sum()),
        "Unique engines": meta.loc[scores.index[valid], "engine"].nunique(),
        "Unscored windows": int((~valid).sum()),
        "AUC": (
            roc_auc_score(labels, values)
            if len(np.unique(labels)) == 2
            else np.nan
        ),
    }
    for limit in (0.05, 0.1, 0.15):
        detection, observed, threshold = genie_roc_point(labels, values, limit)
        suffix = "{:.0%}".format(limit)
        row["".join(
            (
                "Detection at ",
                "{}".format(suffix),
                " control-window flags",
            ),
        )] = detection
        row["".join(
            (
                "Observed control-window flags at ",
                "{}".format(suffix),
            ),
        )] = observed
        row["".join(
            ("Exploratory threshold at ", "{}".format(suffix)),
        )] = threshold
    audit = (
        (
            (
                predictions.groupby("sample_id", sort=False)["audit_score"]
            ).mean()
        ).dropna()
        if "audit_score" in predictions
        else pd.Series(dtype=float)
    )
    audit_labels = meta.loc[audit.index, "_label"]
    row["Date/history/coverage-only AUC"] = (
        roc_auc_score(audit_labels, audit)
        if audit_labels.nunique() == 2
        else np.nan
    )
    return row


def genie_cv_scores(
    frame, head, config, settings, stage="Genie reproduction",
    grouped=False, training_only=False, repeats=None, progress=print
):
    chosen = eligible_head(frame, head)
    if chosen.empty or chosen["_label"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    counts = (
        chosen.groupby("engine")["_label"].max().value_counts()
        if grouped else chosen["_label"].value_counts()
    )
    if counts.min() < 5:
        return pd.DataFrame(), pd.DataFrame()
    count = int(repeats or settings["genie_repeats"])
    split_count = 5
    if grouped:
        groups = chosen.groupby("engine")["_label"].max().reset_index()
        splits = []
        for repeat in range(count):
            splitter = StratifiedGroupKFold(
                n_splits=split_count, shuffle=True,
                random_state=settings["genie_seed"] + repeat,
            )
            for fold, (train_index, test_index) in enumerate(
                splitter.split(groups, groups["_label"], groups["engine"])
            ):
                learning = set(groups.iloc[train_index]["engine"])
                held = set(groups.iloc[test_index]["engine"])
                splits.append(
                    (
                        repeat, fold,
                        np.flatnonzero(chosen["engine"].isin(learning)),
                        np.flatnonzero(chosen["engine"].isin(held)),
                    )
                )
    else:
        splitter = RepeatedStratifiedKFold(
            n_splits=split_count, n_repeats=count, random_state=settings["genie_seed"]
        )
        splits = [
            (index // split_count, index % split_count, learning, held)
            for index, (learning, held) in enumerate(splitter.split(chosen, chosen["_label"]))
        ]
    signature = runtime_frame_signature(frame)
    identity = (
        signature, head, stage, grouped, training_only, count,
        runtime_scientific_settings(settings),
        tuple(P15["genie_keep"]) if not training_only else None,
    )
    state = runtime_checkpoint("cv", identity, tuple(config))
    completed = state.setdefault("folds", {})
    audits = runtime_checkpoint("date_audit", identity)
    columns = (
        genie_feature_columns(frame) if training_only else list(P15["genie_keep"])
    )
    source = chosen.copy()
    if not training_only:
        source[columns] = source[columns].fillna(frame[columns].median())
    predictions, checks = [], []
    for repeat, fold, learning, held in splits:
        key = (repeat, fold)
        label = "{} {} {}/K={} | repeat {}/{} fold {}/{}".format(
            stage, head, config[0], config[1], repeat + 1, count, fold + 1, split_count
        )
        if key in completed:
            packet = completed[key]
            predictions.append(packet["output"])
            checks.append(packet["check"])
            runtime_message(progress, "REUSE {}".format(label))
            continue
        with runtime_operation(label, progress):
            train, test = source.iloc[learning], source.iloc[held]
            overlap = set(train["engine"]) & set(test["engine"])
            if grouped and overlap:
                raise AssertionError("Grouped reconstruction has engine identity overlap")
            output = test[["sample_id", "engine", "category", "cutoff", "_label"]].copy()
            output["audit_score"] = np.nan
            reason, selected_count, constant_count, warning_count = "", 0, 0, 0
            try:
                model = genie_model_fit(train, config, columns, training_only)
                runtime_model_notes(model, progress)
                prediction = genie_predict(model, test, settings, assess=training_only)
                output["score"] = prediction["score"]
                selected_count = len(model["selected"])
                constant_count = model.get("constant_feature_count", 0)
                warning_count = len(model.get("fitting_warnings", []))
            except ValueError as exc:
                reason = str(exc)
                output["score"] = np.nan
                runtime_message(progress, "UNASSESSED model fold: {}".format(reason))
            if key in audits:
                output["audit_score"] = audits[key]
            else:
                try:
                    audit = genie_date_audit(train, test)
                    output["audit_score"] = audit
                    audits[key] = np.asarray(audit).copy()
                except ValueError as exc:
                    runtime_message(progress, "Date/coverage audit unavailable: {}".format(exc))
            output["repeat"], output["fold"] = repeat + 1, fold + 1
            check = {
                "Experiment": stage, "Issue": head, "Repeat": repeat + 1,
                "Fold": fold + 1, "Overlapping train/test engines": len(overlap),
                "Selected features": selected_count, "Constant features": constant_count,
                "Recorded training warnings": warning_count, "Model reason": reason,
            }
            completed[key] = {"output": output, "check": check}
        predictions.append(output)
        checks.append(check)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(checks)


def run_genie_benchmark(settings, progress=print):
    frame = P15["genie_matrix"]
    rows, scores, checks, best = [], {}, [], {}

    def publish():
        P15["benchmark_results"] = pd.DataFrame(rows)
        P15["benchmark_scores"] = dict(scores)
        P15["benchmark_best"] = dict(best)
        P15["benchmark_fold_checks"] = (
            pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
        )

    grid = [(kind, k, "all") for kind in ("gbm", "rf", "lr") for k in (20, 50, 100)]
    for head in ("HPT1", "TRU", "ANY"):
        candidates = grid if settings["benchmark_grid"] else [GENIE_PRESETS[head]]
        for index, config in enumerate(candidates, 1):
            runtime_message(
                progress, "Genie reproduction {}: candidate {}/{} starting".format(
                    head, index, len(candidates)
                )
            )
            prediction, folds = genie_cv_scores(
                frame, head, config, settings, progress=progress
            )
            if prediction.empty:
                runtime_message(progress, "UNASSESSED candidate: insufficient classes or windows")
                continue
            profile = roc_profile(prediction, "Genie reproduction", head, config)
            rows.append(profile)
            scores[head, config] = prediction
            checks.append(folds)
            rank = profile["Detection at 10% control-window flags"], profile["AUC"]
            if pd.notna(rank[0]) and (head not in best or rank > best[head]["rank"]):
                best[head] = {"config": config, "rank": rank, "profile": profile}
            publish()
            runtime_message(
                progress, "Genie reproduction {}: candidate {}/{} complete; results retained".format(
                    head, index, len(candidates)
                )
            )
    publish()
    config = ("rf_bal", 144, "tru")
    runtime_message(
        progress,
        "Additional TRU subsystem benchmark starting: {} folds, 500 trees per fit".format(
            5 * settings["tru_repeats"]
        ),
    )
    prediction, folds = genie_cv_scores(
        frame, "TRU", config, {**settings, "genie_seed": settings["tru_seed"]},
        stage="Genie later TRU subsystem search", repeats=settings["tru_repeats"],
        progress=progress,
    )
    if not prediction.empty:
        rows.append(roc_profile(prediction, "Genie later TRU subsystem search", "TRU", config))
        scores["TRU", config] = prediction
        checks.append(folds)
    else:
        runtime_message(progress, "UNASSESSED additional TRU benchmark")
    publish()
    return P15["benchmark_results"]


def run_reconstruction_comparisons(settings, progress=print):
    rows, checks = [], []
    experiments = [
        ("Engine separation; original dates", P15["genie_matrix"], False),
        ("Training-only preprocessing; original dates", P15["genie_matrix"], True),
        ("Matched dates; same fixed candidates", P15["cohort"], True),
    ]
    for stage, frame, training_only in experiments:
        for head in ("HPT1", "TRU", "ANY"):
            candidates = [GENIE_PRESETS[head]]
            if AUDIT_PRESETS[head] not in candidates:
                candidates.append(AUDIT_PRESETS[head])
            for config in candidates:
                prediction, folds = genie_cv_scores(
                    frame, head, config, settings, stage, True, training_only,
                    progress=progress,
                )
                if not prediction.empty:
                    rows.append(roc_profile(prediction, stage, head, config))
                    checks.append(folds)
                P15["comparison_results"] = pd.DataFrame(rows)
                P15["comparison_fold_checks"] = (
                    pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
                )
            runtime_message(
                progress, "Reconstruction check: {} | {} complete".format(stage, head)
            )
    P15["comparison_results"] = pd.DataFrame(rows)
    P15["comparison_fold_checks"] = (
        pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    )
    return P15["comparison_results"]


def genie_date_audit(train, test):

    def values(frame):
        out = pd.DataFrame(index=frame.index)
        out["cutoff_days"] = frame["cutoff"].map(
            lambda stamp: stamp.value / 86400000000000.0,
        )
        for name in frame:
            if name.startswith("quality|"):
                out[name] = pd.to_numeric(frame[name], errors="coerce")
        out["missing_fraction"] = frame[genie_feature_columns(
            frame)].isna().mean(axis=1)
        return out
    left, right = (values(train), values(test))
    keep = [name for name in left if left[name].notna().any()]
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ],
    )
    estimator.fit(left[keep], train["_label"])
    return estimator.predict_proba(right.reindex(columns=keep))[:, 1]


def validate_reconstructed_head(cohort, head, settings, progress=print):
    chosen = eligible_head(cohort, head)
    result = {
        "head": head,
        "status": "Classifier not established",
        "metrics": {},
        "test": pd.DataFrame(),
        "all_predictions": pd.DataFrame(),
        "repeat_metrics": pd.DataFrame(),
        "folds": pd.DataFrame(),
        "models": {},
        "warnings": [],
    }
    counts = (
        chosen.groupby("engine")["_label"].max().value_counts()
    ).reindex(
        [0, 1],
        fill_value=0,
    )
    if counts[1] < 6 or counts[0] < 12:
        result["warnings"] = [
            "".join(
                (
                    "{}".format(counts[1]),
                    " independent incident engines and ",
                    "{}".format(counts[0]),
                    (
                        " control engines are eligible"
                        "; 6 and 12 are required. Indi"
                        "vidual EPS/raw reviews remain"
                        " independent."
                    ),
                ),
            ),
        ]
        return result
    config = AUDIT_PRESETS[head]
    state = runtime_checkpoint(
        "audit_folds", runtime_frame_signature(cohort), head,
        runtime_scientific_settings(settings),
    )
    completed = state.setdefault("folds", {})
    engines = chosen.groupby("engine")["_label"].max().reset_index()
    all_predictions, checks, repeated = ([], [], [])
    P15["event_models"][head] = {}
    for repeat in range(settings["audit_repeats"]):
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=settings["genie_seed"] + repeat,
        )
        outputs = []
        for fold, (learning, held) in enumerate(
            splitter.split(
                engines,
                engines["_label"],
                engines["engine"],
            ),
        ):
            fold_key = (repeat, fold)
            label = "Calibrated {} validation | repeat {}/{} fold {}/5".format(
                head, repeat + 1, settings["audit_repeats"], fold + 1
            )
            if fold_key in completed:
                packet = completed[fold_key]
                output, model = packet["output"], packet["model"]
                checks.append(packet["check"])
                outputs.append(output)
                if model is not None:
                    result["models"][fold_key] = model
                    if repeat == 0:
                        for event_id in output.loc[output["_label"].eq(1), "sample_id"]:
                            P15["event_models"][head][event_id] = model
                runtime_message(progress, "REUSE {}".format(label))
                continue
            with runtime_operation(label, progress):
                pool = chosen.loc[chosen["engine"].isin(
                    set(engines.iloc[learning]["engine"]),
                )]
                test = chosen.loc[chosen["engine"].isin(
                    set(engines.iloc[held]["engine"]),
                )]
                controls = sorted(
                    pool.loc[pool["_label"].eq(0), "engine"].unique(),
                    key=lambda engine: (
                        hashlib.sha256(
                            (
                                "".join(
                                    (
                                        "{}".format(repeat),
                                        ":",
                                        "{}".format(fold),
                                        ":",
                                        "{}".format(engine),
                                    ),
                                )
                            ).encode(),
                        )
                    ).hexdigest(),
                )
                n = min(
                    max(10, int(np.ceil(len(controls) * 0.25))),
                    max(0, len(controls) - 5),
                )
                calibration_engines = set(controls[:n])
                train = pool.loc[~pool["engine"].isin(calibration_engines)]
                calibration = pool.loc[pool["engine"].isin(calibration_engines)]
                if (
                    set(train["engine"]) & set(test["engine"])
                    or calibration_engines & set(test["engine"])
                    or set(train["engine"]) & calibration_engines
                ):
                    raise AssertionError(
                        (
                            "Training, calibration and hel"
                            "d-out engine identities overl"
                            "ap"
                        ),
                    )
                output = test.copy()
                (
                    output["score"],
                    output["audit_score"],
                    output["alert_threshold"],
                ) = (np.nan, np.nan, np.nan)
                output["assessed"], output["alert_assessed"] = (False, False)
                output["flag"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
                output["reason"] = "Model not fitted"
                threshold, independent = (np.inf, 0)
                try:
                    model = genie_model_fit(train, config)
                    runtime_model_notes(model, progress)
                    cal_prediction = genie_predict(model, calibration, settings)
                    cal = pd.concat([calibration, cal_prediction], axis=1)
                    threshold, independent, reason = calibrated_threshold(
                        cal.loc[cal["assessed"]],
                        settings["false_alarm_limit"],
                    )
                    prediction = genie_predict(model, test, settings)
                    output[["score", "assessed", "reason"]
                           ] = prediction[["score", "assessed", "reason"]]
                    output["audit_score"] = genie_date_audit(train, test)
                    output["alert_threshold"] = threshold
                    output["alert_assessed"] = output["assessed"] & np.isfinite(threshold)
                    output.loc[output["alert_assessed"], "flag"] = (
                        output.loc[output["alert_assessed"], "score"].gt(
                            threshold,
                        )
                    )
                    output.loc[output["assessed"] & ~
                               output["alert_assessed"], "reason"] = reason
                    model["calibration_engines"] = calibration_engines
                    result["models"][repeat, fold] = model
                    if repeat == 0:
                        for event_id in test.loc[test["_label"].eq(1), "sample_id"]:
                            P15["event_models"][head][event_id] = model
                except ValueError as exc:
                    output["reason"] = str(exc)
                output["repeat"], output["fold"] = (repeat + 1, fold + 1)
                checks.append(
                    {
                        "Repeat": repeat + 1,
                        "Fold": fold + 1,
                        "Training engines": train["engine"].nunique(),
                        "Calibration control engines": independent,
                        "Held-out engines": test["engine"].nunique(),
                        "Model": config[0],
                        "Threshold": threshold,
                        "Reason": "; ".join(
                            output["reason"].drop_duplicates().astype(
                                str,
                            ),
                        ),
                    },
                )
                outputs.append(output)
                completed[fold_key] = {
                    "output": output, "check": checks[-1],
                    "model": result["models"].get(fold_key),
                }
        predictions = pd.concat(outputs, ignore_index=True)
        if (
            predictions["sample_id"].duplicated().any()
            or (
                set(predictions["sample_id"])
                != set(chosen["sample_id"])
            )
        ):
            raise AssertionError(
                (
                    "Every eligible window must be hel"
                    "d out once per repeat"
                ),
            )
        all_predictions.append(predictions)
        repeated.append(
            {
                "Repeat": repeat + 1,
                **cross_validation_metrics(predictions, settings),
            },
        )
        progress(
            "".join(
                (
                    "Matched-date engine-held-out assessment ",
                    "{}".format(head),
                    ": repeat ",
                    "{}".format(repeat + 1),
                    "/",
                    "{}".format(settings["audit_repeats"]),
                    " complete",
                ),
            ),
        )
    primary = all_predictions[0]
    metrics = cross_validation_metrics(primary, settings)
    notices = [
        (
            "Retrospective assessment; documented cand"
            "idate settings were chosen in prior fleet"
            " experiments. Independent prospective per"
            "formance remains unverified."
        ),
        (
            "Controls have no documented incidents; he"
            "althy status is unverified. Control flags"
            " are not a confirmed healthy-engine FPR."
        ),
        (
            "EPS record features are independent of 10"
            "0\u2013300-anchor parameter reviews. EPS "
            "meanings and earlier-only generation requ"
            "ire engineering confirmation."
        ),
        (
            "Matched dates, phase availability, histor"
            "y length and recency are checked; equival"
            "ent operating conditions are not fully ve"
            "rified."
        ),
    ]
    if metrics["Date/coverage-only AUC"] >= 0.7:
        notices.append(
            (
                "Date/history/coverage still distingui"
                "shes labels; confounding remains unre"
                "solved."
            ),
        )
    if metrics["Unassessed test windows"]:
        notices.append(
            "".join(
                (
                    "{}".format(
                        metrics["Unassessed test windows"],
                    ),
                    (
                        " windows lack selected EPS ev"
                        "idence or a sufficiently cali"
                        "brated threshold."
                    ),
                ),
            ),
        )
    result.update(
        {
            "status": (
                "Retrospective classifier assessed"
                if primary["alert_assessed"].any()
                else "Scores available; threshold not established"
            ),
            "metrics": metrics,
            "test": primary,
            "all_predictions": pd.concat(all_predictions, ignore_index=True),
            "repeat_metrics": pd.DataFrame(repeated),
            "folds": pd.DataFrame(checks),
            "warnings": notices,
        },
    )
    return result


def run_reconstructed_validation(settings, progress=print):
    state = runtime_checkpoint(
        "validated_heads", runtime_frame_signature(P15["cohort"]),
        runtime_scientific_settings(settings),
    )
    completed = state.setdefault("heads", {})
    results = {}
    for head in ("HPT1", "HPT2", "TRU", "ANY"):
        if head in completed:
            result = completed[head]
            runtime_message(progress, "REUSE completed calibrated head: {}".format(head))
            model_map = P15["event_models"].setdefault(head, {})
            for (repeat, fold), model in result["models"].items():
                if repeat == 0:
                    test = result["test"]
                    for event_id in test.loc[
                        test["fold"].eq(fold + 1) & test["_label"].eq(1), "sample_id"
                    ]:
                        model_map[event_id] = model
        else:
            result = validate_reconstructed_head(P15["cohort"], head, settings, progress)
            completed[head] = result
        results[head] = result
        P15["validation"][head] = result
    P15["validation"] = results
    P15["manifest"] = {
        "version": NOTEBOOK_VERSION,
        "settings": settings,
        "tables": P15.get("table_names", TABLE_NAMES),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": (
            "Code from Genie.pdf: final build_feats, p"
            "ick_cols, repeated-CV grid and EPS-only T"
            "RU subsystem experiment"
        ),
        "reproduction": (
            "One row per documented event plus one per"
            " control engine; global coverage/median p"
            "reprocessing and row-stratified CV reprod"
            "uced as exploratory benchmark"
        ),
        "audit": (
            "Engine-held-out, training-only preprocess"
            "ing, matched dates and independent calibr"
            "ation; primary alert metrics use first re"
            "peat"
        ),
        "reference_counts": {
            "target_rows": 320,
            "event_rows": 81,
            "control_engines": 239,
            "features": 501,
            "level": 253,
            "change": 248,
        },
        "actual_counts": {
            "benchmark_rows": len(P15["genie_matrix"]),
            "benchmark_unique_engines": P15["genie_matrix"]["engine"].nunique(),
            "benchmark_features": len(P15["genie_keep"]),
            "audit_windows": len(P15["cohort"]),
        },
        "limitations": (
            "No claim of reproduced percentages until "
            "fleet execution. EPS flags may contain ag"
            "e/coverage effects; nominal references ar"
            "e excluded from measured outputs."
        ),
    }
    return results


def parameter_evidence(model, window):
    observation = pd.DataFrame([window])
    settings = P15["settings"]
    base = genie_predict(model, observation, settings)
    if not bool(base.iloc[0]["assessed"]):
        return pd.DataFrame()
    groups = {}
    for name in model["selected"]:
        prefix, signal, statistic = name.split(":", 2)
        phase = next(
            (
                phase
                for phase, short in GENIE_PREFIX.items()
                if short == prefix
            ),
        )
        (
            groups.setdefault((phase, signal_family(signal)), [])
        ).append(
            name,
        )
    rows = []
    for (phase, family), names in groups.items():
        counter = observation.copy()
        for name in names:
            counter[name] = model["control_center"].get(name, np.nan)
        alternate = (
            genie_predict(model, counter, settings, assess=False)
        ).iloc[0]["score"]
        driver = max(
            names,
            key=lambda name: (
                abs(
                    float(
                        (
                            observation.iloc[0].get(name, np.nan)
                            - model["control_center"].get(
                                name,
                                np.nan,
                            )
                        ),
                    ),
                )
                if (
                    pd.notna(observation.iloc[0].get(name))
                    and pd.notna(model["control_center"].get(name))
                )
                else -1
            ),
        )
        rows.append(
            {
                "Phase": phase,
                "Parameter family": family,
                "EPS feature": driver.split(":")[1],
                "Window statistic": (
                    "Recent level"
                    if driver.endswith(":m")
                    else "Recent minus prior mean"
                ),
                "Score reduction on group replacement": float(base.iloc[0]["score"] - alternate),
                "Interpretation": (
                    "Held-out model sensitivity; not e"
                    "stablished physical causation"
                ),
            },
        )
    return (
        pd.DataFrame(rows).sort_values(
            "Score reduction on group replacement",
            ascending=False,
        )
    ).reset_index(
        drop=True,
    )
