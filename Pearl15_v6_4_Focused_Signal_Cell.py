from scipy.optimize import linear_sum_assignment

P15_SIGNAL_VERSION = "6.4-FOCUSED-SIGNAL"
P15_SIGNAL_OPTIONS = {
    "heads": ("HPT1", "TRU"),
    "lead_days": (0, 7),
    "controls_per_incident": 4,
    "max_reference_records": 300,
    "min_reference_records": 20,
    "include_operating_residuals": True,
    "require_recent_eps": True,
    "outer_repeats": 1,
    "outer_folds": 5,
    "inner_folds": 3,
    "seed": 82719,
    "max_selected_missing": 0.5,
    "resume": True,
}


def p15s_configuration(options=None):
    config = {**P15_SIGNAL_OPTIONS, **(options or {})}
    if set(config) != set(P15_SIGNAL_OPTIONS):
        raise ValueError("Unknown focused-analysis option")
    config["heads"] = tuple(config["heads"])
    config["lead_days"] = tuple(config["lead_days"])
    if not config["heads"] or any(head not in {"HPT1", "TRU"} for head in config["heads"]):
        raise ValueError("heads must contain HPT1 and/or TRU")
    if len(set(config["heads"])) != len(config["heads"]):
        raise ValueError("Duplicate issue requested")
    if not config["lead_days"] or len(set(config["lead_days"])) != len(config["lead_days"]):
        raise ValueError("Use distinct pre-event lead periods")
    integers = {
        "controls_per_incident": (1, 12),
        "max_reference_records": (40, 300),
        "min_reference_records": (20, 100),
        "outer_repeats": (1, 3),
        "outer_folds": (3, 5),
        "inner_folds": (2, 3),
        "seed": (0, 2147483647),
    }
    for name, bounds in integers.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not bounds[0] <= value <= bounds[1]:
            raise ValueError("Invalid focused-analysis option: {}".format(name))
    if any(isinstance(day, bool) or not isinstance(day, (int, np.integer)) or not 0 <= day <= 90 for day in config["lead_days"]):
        raise ValueError("lead_days must contain whole days from 0 to 90")
    for name in ("include_operating_residuals", "require_recent_eps", "resume"):
        if type(config[name]) is not bool:
            raise ValueError("{} must be True or False".format(name))
    if not 0 <= float(config["max_selected_missing"]) <= 0.5:
        raise ValueError("max_selected_missing must be from zero to 0.5")
    if config["min_reference_records"] > config["max_reference_records"]:
        raise ValueError("Minimum reference records exceed maximum")
    return config


def p15s_order(seed, *parts):
    return hashlib.sha256(":".join(map(str, (seed,) + parts)).encode()).hexdigest()


def p15s_eps_slice(engine, phase, cutoff, settings):
    frame = P15["eps_groups"].get(phase, {}).get(engine)
    if frame is None or frame.empty:
        return pd.DataFrame()
    times = P15.get("eps_time_index", {}).get((phase, engine))
    if times is None:
        times = frame["_time"].astype("datetime64[ns, UTC]").array.asi8
    upper = cutoff
    if settings.get("history_end"):
        upper = min(upper, date_boundary(settings["history_end"], settings) + pd.Timedelta(days=1))
    left = 0
    if settings.get("history_start"):
        left = int(np.searchsorted(times, date_boundary(settings["history_start"], settings).value, side="left"))
    right = int(np.searchsorted(times, upper.value, side="left"))
    sub = frame.iloc[left:right]
    if settings.get("strict_asof") and not sub.empty:
        sub = sub.loc[sub["_available"].notna() & sub["_available"].lt(cutoff)]
    return sub


def p15s_phases(window, settings, config):
    selected = []
    for phase in settings["phases"]:
        prefix = "quality|{}|".format(phase)
        count = window.get(prefix + "records", 0)
        gap = window.get(prefix + "gap_days", np.nan)
        coverage = window.get(prefix + "coverage", 0)
        if count >= 5 and pd.notna(coverage) and coverage > 0 and (not config["require_recent_eps"] or pd.notna(gap) and gap <= settings["max_gap_days"]):
            selected.append(phase)
    return tuple(selected)


def p15s_mask_window(window, phases):
    out = dict(window)
    for phase, prefix in GENIE_PREFIX.items():
        if phase not in phases:
            out = {name: value for name, value in out.items() if not name.startswith(prefix + ":")}
            for name in ("records", "coverage", "gap_days", "first_age_days"):
                out["quality|{}|{}".format(phase, name)] = 0 if name in {"records", "coverage"} else np.nan
    return out


def p15s_first_events(head, settings):
    chosen = {}
    for event in sorted(
        (event for event in EVENTS if event_category(event) == head),
        key=lambda item: (date_boundary(item["date"], settings), str(item["id"])),
    ):
        chosen.setdefault(canonical_engine(event["esn"]), event)
    return list(chosen.values())


def p15s_matched_engines(head, lead, settings, config, state, progress=print):
    key = runtime_key("matching", head, lead)
    if key in state.setdefault("cohorts", {}):
        runtime_message(progress, "REUSE issue-specific {} cohort at {} days".format(head, lead))
        return state["cohorts"][key]
    events = p15s_first_events(head, settings)
    documented_engines = {canonical_engine(event["esn"]) for event in EVENTS}
    controls = sorted(set(P15["control_engines"]) - documented_engines)
    cache = state.setdefault("matching_cases", {})
    packets, unavailable = [], []
    for index, event in enumerate(events, 1):
        packet_key = (head, lead, str(event["id"]))
        if packet_key in cache:
            packet = cache[packet_key]
            runtime_message(progress, "REUSE matching {}/{} {} {} days".format(index, len(events), head, lead))
        else:
            cutoff = date_boundary(event["date"], settings) - pd.Timedelta(days=lead)
            positive = genie_snapshot_window(event["esn"], cutoff, settings, head, event["id"])
            phases = p15s_phases(positive, settings, config)
            candidates = []
            label = "{} {} days: match incident engine {}/{}".format(head, lead, index, len(events))
            with runtime_operation(label, progress):
                if phases:
                    positive = p15s_mask_window(positive, phases)
                    for position, engine in enumerate(controls, 1):
                        ends = [P15["eps_groups"][phase][engine]["_time"].iloc[-1] for phase in phases if engine in P15["eps_groups"][phase] and not P15["eps_groups"][phase][engine].empty]
                        if ends and max(ends) >= cutoff + pd.Timedelta(days=settings["horizon_days"]):
                            negative = genie_snapshot_window(engine, cutoff, settings, include_features=False)
                            if set(phases).issubset(p15s_phases(negative, settings, config)):
                                distance = matching_distance(positive, p15s_mask_window(negative, phases))
                                if np.isfinite(distance) and distance <= settings["eps_match_distance"]:
                                    candidates.append((engine, float(distance)))
                        if position % 100 == 0 or position == len(controls):
                            runtime_message(progress, "{}: checked {}/{} controls; {} matches".format(label, position, len(controls), len(candidates)))
                packet = {"event": event, "positive": positive, "phases": phases, "candidates": candidates}
                cache[packet_key] = packet
        if packet["phases"] and packet["candidates"]:
            packets.append(packet)
        else:
            unavailable.append({"Issue": head, "Lead days": lead, "Engine": canonical_engine(event["esn"]), "Event": event["id"], "Reason": "No recent pre-event EPS phase or no comparable control with follow-up"})
    remaining = set(controls)
    allocated = [[] for packet in packets]
    for round_number in range(config["controls_per_incident"]):
        active = [i for i in range(len(packets)) if round_number == 0 or allocated[i]]
        engines = sorted(remaining)
        if not active or not engines:
            break
        cost = np.full((len(active), len(engines) + len(active)), 1000000.0)
        positions = {engine: i for i, engine in enumerate(engines)}
        for row_number, packet_index in enumerate(active):
            packet = packets[packet_index]
            for engine, distance in packet["candidates"]:
                if engine in positions:
                    tie = int(p15s_order(config["seed"], head, lead, packet["event"]["id"], engine)[:10], 16) / 16 ** 10
                    cost[row_number, positions[engine]] = distance + tie * 1e-8
        left, right = linear_sum_assignment(cost)
        for row_number, column in zip(left, right):
            if column < len(engines) and cost[row_number, column] < 1000000:
                engine = engines[column]
                allocated[active[row_number]].append(engine)
                remaining.remove(engine)
    rows = []
    for packet, engines in zip(packets, allocated):
        event = packet["event"]
        if not engines:
            unavailable.append({"Issue": head, "Lead days": lead, "Engine": canonical_engine(event["esn"]), "Event": event["id"], "Reason": "No independent control could be allocated to this incident date"})
            continue
        positive = dict(packet["positive"])
        positive.update({"_label": 1, "match_id": str(event["id"]), "match_issue": head, "signal_phases": packet["phases"], "lead_days": lead})
        rows.append(positive)
        for engine in engines:
            negative = genie_snapshot_window(engine, positive["cutoff"], settings)
            negative = p15s_mask_window(negative, packet["phases"])
            negative.update({"_label": 0, "match_id": str(event["id"]), "match_issue": head, "signal_phases": packet["phases"], "lead_days": lead})
            rows.append(negative)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if frame["engine"].duplicated().any() or frame["sample_id"].duplicated().any():
            raise AssertionError("A focused cohort must contain one assessment per independent engine")
        if set(frame.loc[frame["_label"].eq(0), "engine"]) & documented_engines:
            raise AssertionError("A documented incident engine entered the control population")
        for match_id, pair in frame.groupby("match_id"):
            if pair["cutoff"].nunique() != 1 or pair["match_issue"].nunique() != 1 or set(pair["_label"]) != {0, 1}:
                raise AssertionError("An issue-specific matched set is inconsistent")
    packet = {"frame": frame, "unavailable": pd.DataFrame(unavailable), "documented_engines": len(events)}
    state["cohorts"][key] = packet
    runtime_message(progress, "{} at {} days: {} incident engines, {} control engines, {} unavailable incident engines".format(head, lead, int(frame["_label"].sum()) if not frame.empty else 0, int(frame["_label"].eq(0).sum()) if not frame.empty else 0, len(unavailable)))
    return packet


def p15s_eps_features(row, settings, config):
    out = {}
    for phase in row["signal_phases"]:
        prefix = GENIE_PREFIX[phase]
        for name, value in row.items():
            if name.startswith(prefix + ":") and pd.notna(value) and np.isfinite(value):
                out["LEVEL|{}|{}".format(phase, name[len(prefix) + 1:])] = float(value)
        sub = p15s_eps_slice(row["engine"], phase, row["cutoff"], settings).tail(config["max_reference_records"] + 20)
        if len(sub) < config["min_reference_records"] + 10:
            continue
        columns = [column for column in P15["eps_columns"].get(phase, []) if column in sub]
        recent = sub.tail(10)[columns].apply(pd.to_numeric, errors="coerce")
        reference = sub.iloc[:-20].tail(config["max_reference_records"])[columns].apply(pd.to_numeric, errors="coerce")
        if len(reference) < config["min_reference_records"]:
            continue
        center = reference.median()
        scale = reference.sub(center).abs().median() * 1.4826
        scale = scale.where(scale > 1e-12, reference.std())
        supported = reference.count().ge(config["min_reference_records"]) & recent.count().ge(5) & scale.gt(1e-12)
        for column in supported.index[supported]:
            z = recent[column].sub(center[column]).div(scale[column]).clip(-25, 25)
            short = sub.tail(3)[column].apply(pd.to_numeric, errors="coerce").sub(center[column]).div(scale[column]).clip(-25, 25)
            values = z.dropna()
            if len(values) < 5:
                continue
            slope = float(np.polyfit(np.flatnonzero(z.notna()), values.to_numpy(), 1)[0]) if z.notna().sum() >= 5 else np.nan
            for statistic, value in {"shift": z.median(), "last3": short.median(), "slope_record": slope, "magnitude": z.abs().quantile(0.9)}.items():
                if pd.notna(value) and np.isfinite(value):
                    out["DRIFT|{}|{}|{}".format(phase, column, statistic)] = float(value)
    return out


def p15s_residual_features(row, settings):
    out, notices = {}, []
    if not P15.get("flight_groups") or not P15.get("engine_data"):
        return out, ["Operating residuals unavailable: reconciled take-off history is missing"]
    try:
        history = selected_history(row["engine"], row["cutoff"], settings)
    except ValueError as exc:
        return out, [str(exc)]
    anchors = history["_flight"].sort_values().drop_duplicates()
    if (row["cutoff"] - anchors.iloc[-1]).total_seconds() / 86400 > settings["max_gap_days"]:
        return out, ["Operating residuals unavailable: latest reconciled take-off anchor is too old"]
    fit_end = int(len(anchors) * 0.6)
    reference_end = int(len(anchors) * 0.8)
    learning = set(anchors.iloc[:fit_end])
    reference = set(anchors.iloc[fit_end:reference_end])
    monitoring = set(anchors.iloc[reference_end:])
    for phase in row["signal_phases"]:
        da = window_rows(phase + " DA", row["engine"], history, row["cutoff"], settings)
        eps = window_rows(phase + " EPS", row["engine"], history, row["cutoff"], settings)
        merged, source_check = merge_measurement_sources(da, eps, settings)
        if merged.empty:
            notices.append("{}: no actual measurement snapshots".format(phase))
            continue
        actual = [column for column in merged if raw_measurement(column) and signal_family(column) in {"P20", "T20", "ALT", "NH", "NL", "RPM", "MN", "P50", "P30", "T30", "OIP", "OIT", "VBHP", "VBLP"}]
        if not actual:
            notices.append("{}: no supported actual measurements".format(phase))
            continue
        numeric = merged[actual].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric["_flight"] = merged["_flight"]
        data = numeric.groupby("_flight", sort=True).median()
        fit = data.loc[data.index.isin(learning)]
        ref = data.loc[data.index.isin(reference)]
        monitor = data.loc[data.index.isin(monitoring)]
        inputs = []
        for family in ("P20", "T20", "ALT", "NH", "NL", "RPM", "MN"):
            choices = [column for column in actual if signal_family(column) == family and fit[column].notna().mean() >= 0.8]
            choices.sort(key=lambda column: (-fit[column].notna().mean(), column))
            if choices and not (family == "RPM" and any(signal_family(column) in {"NH", "NL"} for column in inputs)):
                inputs.append(choices[0])
        if not inputs:
            notices.append("{}: operating inputs missing in the historical learning period".format(phase))
            continue
        families = ("P50", "P30", "T30", "OIP", "OIT", "VBHP", "VBLP")
        for family in families:
            choices = [column for column in actual if signal_family(column) == family and fit[column].notna().mean() >= 0.8]
            choices.sort(key=lambda column: (-fit[column].notna().mean(), column))
            if not choices:
                continue
            target = choices[0]
            columns = list(dict.fromkeys(inputs + [target]))
            train = fit[columns].dropna()
            calibration = ref[columns].dropna()
            recent = monitor[columns].dropna()
            if len(train) < 30 or len(calibration) < 10 or len(recent) < 5 or train[target].std() <= 1e-12:
                notices.append("{} {}: insufficient varying measurements in chronological stages".format(phase, target))
                continue
            with threadpool_limits(limits=2):
                estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]).fit(train[inputs], train[target])
                cal_domain = operating_domain(calibration, train, inputs)
                recent_domain = operating_domain(recent, train, inputs)
                calibration = calibration.loc[cal_domain]
                recent = recent.loc[recent_domain]
                if len(calibration) < 10 or len(recent) < 5:
                    notices.append("{} {}: insufficient measurements inside the learned operating domain".format(phase, target))
                    continue
                residual = calibration[target].to_numpy() - estimator.predict(calibration[inputs])
                center = float(np.median(residual))
                scale = float(np.median(np.abs(residual - center)) * 1.4826)
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = float(np.std(residual, ddof=1))
                if not np.isfinite(scale) or scale <= 1e-12:
                    notices.append("{} {}: historical residual variation is undefined".format(phase, target))
                    continue
                z = pd.Series((recent[target].to_numpy() - estimator.predict(recent[inputs]) - center) / scale, index=recent.index).clip(-25, 25)
            slope = float(np.polyfit(np.arange(len(z)), z.to_numpy(), 1)[0])
            for statistic, value in {"median": z.median(), "last3": z.tail(3).median(), "slope_anchor": slope, "magnitude": z.abs().quantile(0.9)}.items():
                if pd.notna(value) and np.isfinite(value):
                    out["PHYS|{}|{}|{}".format(phase, target, statistic)] = float(value)
    return out, notices


def p15s_feature_matrix(packet, settings, config, state, progress=print):
    frame = packet["frame"]
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    rows, notices = [], []
    saved = state.setdefault("features", {})
    for position, row in enumerate(frame.to_dict("records"), 1):
        key = runtime_key(row["engine"], row["cutoff"].isoformat(), row["signal_phases"])
        if key in saved:
            features, reasons = saved[key]
            runtime_message(progress, "REUSE focused features {}/{}".format(position, len(frame)))
        else:
            with runtime_operation("Focused features {}/{} engine {}".format(position, len(frame), row["engine"]), progress):
                features = p15s_eps_features(row, settings, config)
                physical, reasons = p15s_residual_features(row, settings) if config["include_operating_residuals"] else ({}, [])
                features.update(physical)
                saved[key] = (features, reasons)
        rows.append({**row, **features})
        notices.extend({"Engine": row["engine"], "Issue": row["match_issue"], "Lead days": row["lead_days"], "Reason": reason} for reason in reasons)
        runtime_message(progress, "FEATURES {}/{}: {} EPS drift and {} operating residual features".format(position, len(frame), sum(name.startswith("DRIFT|") for name in features), sum(name.startswith("PHYS|") for name in features)))
    return pd.DataFrame(rows), pd.DataFrame(notices)


def p15s_candidates(head):
    candidates = [(view, kind, 20 if kind == "lr" else 30) for view in ("levels", "changes", "combined") for kind in ("lr", "rf")]
    if head == "TRU":
        candidates.extend(("combined_tru", kind, 20 if kind == "lr" else 30) for kind in ("lr", "rf"))
    return candidates


def p15s_feature_columns(frame, view):
    prefixes = {"levels": ("LEVEL|",), "changes": ("DRIFT|", "PHYS|"), "combined": ("LEVEL|", "DRIFT|", "PHYS|"), "combined_tru": ("LEVEL|", "DRIFT|", "PHYS|")}
    columns = [column for column in frame if column.startswith(prefixes[view])]
    if view == "combined_tru":
        columns = [column for column in columns if any(pattern in column.upper() for pattern in TRU_PATTERNS)]
    return columns


def p15s_fit(train, candidate, config):
    view, kind, topk = candidate
    columns = p15s_feature_columns(train, view)
    if not columns or train["_label"].nunique() < 2:
        raise ValueError("No supported signal features or insufficient label classes")
    numeric = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = [column for column in columns if numeric[column].notna().mean() >= 0.6 and pd.notna(numeric[column].std()) and numeric[column].std() > 1e-12]
    if not keep:
        raise ValueError("No varying training features with adequate coverage")
    median = numeric[keep].median()
    filled = numeric[keep].fillna(median)
    keep = [column for column in keep if filled[column].max() - filled[column].min() > 1e-12]
    if not keep:
        raise ValueError("All imputed training features are constant")
    values = filled[keep].to_numpy(dtype=float)
    labels = train["_label"].to_numpy(dtype=int)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        scores, unused = f_classif(values, labels)
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    order = sorted(range(len(keep)), key=lambda index: (-scores[index], keep[index]))[:min(topk, len(keep))]
    selected = [keep[index] for index in order]
    scaler = StandardScaler().fit(filled[selected])
    if kind == "lr":
        estimator = LogisticRegression(C=0.2, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=config["seed"])
    else:
        estimator = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=3, class_weight="balanced_subsample", max_features="sqrt", n_jobs=2, random_state=config["seed"])
    with threadpool_limits(limits=2):
        estimator.fit(scaler.transform(filled[selected]), labels)
    importance = np.abs(estimator.coef_[0]) if kind == "lr" else estimator.feature_importances_
    importance = importance / importance.sum() if importance.sum() > 0 else importance
    return {"candidate": candidate, "selected": selected, "median": median[selected], "scaler": scaler, "estimator": estimator, "importance": dict(zip(selected, map(float, importance))), "training_engines": set(train["engine"]), "warning_count": len(captured)}


def p15s_predict(model, frame, config):
    numeric = frame.reindex(columns=model["selected"]).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().mean(axis=1).ge(1 - config["max_selected_missing"])
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.any():
        with threadpool_limits(limits=2):
            scores.loc[valid] = model["estimator"].predict_proba(model["scaler"].transform(numeric.loc[valid].fillna(model["median"])))[:, 1]
    return scores


def p15s_inner_choice(train, head, config, checkpoint, progress=print):
    if train["engine"].duplicated().any():
        raise AssertionError("Inner selection received repeated engine assessments")
    folds = min(config["inner_folds"], int(train["_label"].value_counts().min()))
    if folds < 2:
        raise ValueError("Insufficient independent engines for inner selection")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config["seed"] + 101)
    splits = list(splitter.split(train, train["_label"], train["engine"]))
    candidates = p15s_candidates(head)
    packets = checkpoint.setdefault("inner", {})
    profiles = []
    for position, candidate in enumerate(candidates, 1):
        candidate_key = tuple(candidate)
        if candidate_key in packets:
            profile = packets[candidate_key]
            runtime_message(progress, "REUSE inner candidate {}/{} {}".format(position, len(candidates), candidate))
        else:
            scored = pd.Series(np.nan, index=train.index, dtype=float)
            reasons = []
            completed = checkpoint.setdefault("inner_folds", {}).setdefault(candidate_key, {})
            for fold, (learning, held) in enumerate(splits):
                if fold in completed:
                    score, reason = completed[fold]
                else:
                    label = "Inner candidate {}/{} {} fold {}/{}".format(position, len(candidates), candidate, fold + 1, folds)
                    with runtime_operation(label, progress):
                        left, right = train.iloc[learning], train.iloc[held]
                        if set(left["engine"]) & set(right["engine"]):
                            raise AssertionError("Inner learning and validation engine identities overlap")
                        try:
                            model = p15s_fit(left, candidate, config)
                            score, reason = p15s_predict(model, right, config), ""
                        except ValueError as exc:
                            score, reason = pd.Series(np.nan, index=right.index), str(exc)
                        completed[fold] = (score, reason)
                scored.loc[score.index] = score
                if reason:
                    reasons.append(reason)
            valid = scored.notna()
            positive = train["_label"].eq(1)
            negative = ~positive
            profile = {"candidate": candidate, "coverage": float(valid.mean()), "incident_coverage": float(valid.loc[positive].mean()), "control_coverage": float(valid.loc[negative].mean()), "selection_detection": 0.0, "AUC": np.nan, "Reason": "; ".join(sorted(set(reasons)))}
            if valid.any() and train.loc[valid, "_label"].nunique() == 2:
                labels = train.loc[valid, "_label"].to_numpy(dtype=int)
                values = scored.loc[valid].to_numpy()
                fpr, tpr, thresholds = roc_curve(labels, values)
                supported = fpr < 0.1
                profile["selection_detection"] = float(np.max(tpr[supported])) * profile["incident_coverage"]
                profile["AUC"] = float(roc_auc_score(labels, values))
            packets[candidate_key] = profile
        profiles.append(profile)
    usable = [profile for profile in profiles if pd.notna(profile["AUC"]) and profile["incident_coverage"] >= 0.8 and profile["control_coverage"] >= 0.8]
    if not usable:
        raise ValueError("No candidate has adequate inner-validation signal coverage")
    selected = max(usable, key=lambda profile: (profile["selection_detection"], profile["AUC"], profile["coverage"], -profile["candidate"][2]))
    runtime_message(progress, "INNER SELECTED {} | development detection {:.1%}, AUC {:.3f}".format(selected["candidate"], selected["selection_detection"], selected["AUC"]))
    return selected["candidate"], profiles


def p15s_metrics(test):
    scored = test.loc[test["score"].notna()]
    evaluated = test.loc[test["flag"].notna()]
    positive = evaluated["_label"].eq(1)
    flags = evaluated["flag"].astype(bool)
    tp, fn = int((positive & flags).sum()), int((positive & ~flags).sum())
    fp, tn = int((~positive & flags).sum()), int((~positive & ~flags).sum())
    incident_total = int(test["_label"].eq(1).sum())
    control_total = int(test["_label"].eq(0).sum())
    detection = tp / (tp + fn) if tp + fn else np.nan
    control_rate = fp / (fp + tn) if fp + tn else np.nan
    detection_low, detection_high = binomial_interval(tp, tp + fn)
    control_low, control_high = binomial_interval(fp, fp + tn)
    audit = test.dropna(subset=["audit_score"])
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "Detection rate": detection, "Control flag rate": control_rate, "Incident engines": incident_total, "Control engines": control_total, "Unassessed incident engines": incident_total - tp - fn, "Unassessed control engines": control_total - fp - tn, "Detected / eligible incident engines": tp / incident_total if incident_total else np.nan, "Assessment coverage": len(evaluated) / len(test) if len(test) else np.nan, "AUC": float(roc_auc_score(scored["_label"], scored["score"])) if scored["_label"].nunique() == 2 else np.nan, "Date/coverage AUC": float(roc_auc_score(audit["_label"], audit["audit_score"])) if audit["_label"].nunique() == 2 else np.nan, "Detection 95% lower": detection_low, "Detection 95% upper": detection_high, "Control flag 95% lower": control_low, "Control flag 95% upper": control_high, "Point target met": bool(incident_total and pd.notna(control_rate) and tp / incident_total >= 0.8 and control_rate < 0.1 and len(evaluated) / len(test) >= 0.9)}


def p15s_validate(frame, head, lead, config, state, progress=print):
    empty = {"status": "Not established", "test": pd.DataFrame(), "all_predictions": pd.DataFrame(), "repeat_metrics": pd.DataFrame(), "folds": pd.DataFrame(), "features": pd.DataFrame(), "inner_profiles": pd.DataFrame(), "metrics": {}}
    if frame.empty:
        return {**empty, "status": "No eligible issue-specific matched engines"}
    if frame["engine"].duplicated().any():
        raise AssertionError("Focused validation requires one assessment per engine")
    counts = frame["_label"].value_counts().reindex([0, 1], fill_value=0)
    if counts[1] < 6 or counts[0] < 20:
        return {**empty, "status": "Only {} incident and {} control engines; at least 6 and 20 are required".format(counts[1], counts[0])}
    data_key = runtime_key(head, lead, runtime_frame_signature(frame))
    checkpoint = state.setdefault("validation", {}).setdefault(data_key, {})
    completed = checkpoint.setdefault("outer_folds", {})
    predictions, checks, selected_features, profiles, repeated = [], [], [], [], []
    for repeat in range(config["outer_repeats"]):
        splitter = StratifiedGroupKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["seed"] + repeat)
        outputs = []
        for fold, (learning, held) in enumerate(splitter.split(frame, frame["_label"], frame["engine"])):
            fold_key = (repeat, fold)
            label = "Focused {} {} days | repeat {}/{} outer fold {}/{}".format(head, lead, repeat + 1, config["outer_repeats"], fold + 1, config["outer_folds"])
            if fold_key in completed:
                packet = completed[fold_key]
                runtime_message(progress, "REUSE {}".format(label))
            else:
                partial = checkpoint.setdefault("partial_folds", {}).setdefault(fold_key, {})
                with runtime_operation(label, progress):
                    pool, test = frame.iloc[learning].copy(), frame.iloc[held].copy()
                    control_engines = sorted(pool.loc[pool["_label"].eq(0), "engine"], key=lambda engine: p15s_order(config["seed"], repeat, fold, engine))
                    ncal = min(max(10, int(np.ceil(len(control_engines) * 0.25))), max(0, len(control_engines) - 6))
                    calibration_engines = set(control_engines[:ncal])
                    calibration = pool.loc[pool["engine"].isin(calibration_engines)]
                    train = pool.loc[~pool["engine"].isin(calibration_engines)]
                    train_engines, test_engines = set(train["engine"]), set(test["engine"])
                    if train_engines & test_engines or calibration_engines & test_engines or train_engines & calibration_engines:
                        raise AssertionError("Training, calibration and held-out engines overlap")
                    out = test.copy()
                    out["score"], out["audit_score"] = np.nan, np.nan
                    out["flag"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
                    out["repeat"], out["fold"] = repeat + 1, fold + 1
                    out["threshold"], out["reason"] = np.nan, "Model not fitted"
                    feature_rows, inner_rows = [], []
                    candidate, independent, threshold = None, 0, np.inf
                    try:
                        candidate, inner = p15s_inner_choice(train, head, config, partial, progress)
                        model = p15s_fit(train, candidate, config)
                        cal = calibration.copy()
                        cal["score"] = p15s_predict(model, calibration, config)
                        threshold, independent, reason = calibrated_threshold(cal, 0.1)
                        out["score"] = p15s_predict(model, test, config)
                        out["threshold"] = threshold
                        usable = out["score"].notna() & np.isfinite(threshold)
                        out.loc[usable, "flag"] = out.loc[usable, "score"].gt(threshold)
                        out["reason"] = "Assessed"
                        out.loc[out["score"].isna(), "reason"] = "Selected measurement evidence is insufficient"
                        out.loc[out["score"].notna() & ~usable, "reason"] = reason
                        try:
                            out["audit_score"] = genie_date_audit(train, test)
                        except ValueError as exc:
                            runtime_message(progress, "Date/coverage audit unavailable: {}".format(exc))
                        feature_rows = [{"Issue": head, "Lead days": lead, "Repeat": repeat + 1, "Fold": fold + 1, "Feature": column, "Importance": model["importance"][column], "Candidate": str(candidate)} for column in model["selected"]]
                        inner_rows = [{"Issue": head, "Lead days": lead, "Repeat": repeat + 1, "Fold": fold + 1, **profile} for profile in inner]
                    except ValueError as exc:
                        out["reason"] = str(exc)
                        runtime_message(progress, "UNASSESSED {}: {}".format(label, exc))
                    check = {"Issue": head, "Lead days": lead, "Repeat": repeat + 1, "Fold": fold + 1, "Training engines": len(train_engines), "Calibration engines": len(calibration_engines), "Scored calibration engines": independent, "Held-out engines": len(test_engines), "Threshold": threshold, "Candidate": str(candidate), "Train engine identities": tuple(sorted(train_engines)), "Calibration engine identities": tuple(sorted(calibration_engines)), "Test engine identities": tuple(sorted(test_engines)), "Reason": "; ".join(out["reason"].astype(str).drop_duplicates())}
                    packet = {"output": out, "check": check, "features": feature_rows, "profiles": inner_rows}
                    completed[fold_key] = packet
            outputs.append(packet["output"])
            checks.append(packet["check"])
            selected_features.extend(packet["features"])
            profiles.extend(packet["profiles"])
        primary = pd.concat(outputs, ignore_index=True)
        if primary["engine"].duplicated().any() or set(primary["engine"]) != set(frame["engine"]):
            raise AssertionError("Each engine must be held out exactly once per repeat")
        predictions.append(primary)
        repeated.append({"Repeat": repeat + 1, **p15s_metrics(primary)})
        runtime_message(progress, "COMPLETE {} {} days repeat {}/{}".format(head, lead, repeat + 1, config["outer_repeats"]))
    primary = predictions[0]
    return {"status": "Development cross-validation assessed" if primary["flag"].notna().any() else "Scores or thresholds not established", "test": primary, "all_predictions": pd.concat(predictions, ignore_index=True), "repeat_metrics": pd.DataFrame(repeated), "folds": pd.DataFrame(checks), "features": pd.DataFrame(selected_features), "inner_profiles": pd.DataFrame(profiles), "metrics": p15s_metrics(primary)}


def p15s_summary_row(head, lead, result, packet):
    metrics = result["metrics"]
    return {"Issue": head, "Lead days": lead, "Documented incident engines": packet["documented_engines"], "Eligible incident engines": metrics.get("Incident engines", int(packet["frame"]["_label"].sum()) if not packet["frame"].empty else 0), "Eligible control engines": metrics.get("Control engines", int(packet["frame"]["_label"].eq(0).sum()) if not packet["frame"].empty else 0), "TP": metrics.get("TP", "Not assessed"), "FN": metrics.get("FN", "Not assessed"), "FP": metrics.get("FP", "Not assessed"), "TN": metrics.get("TN", "Not assessed"), "Detection": percent_text(metrics.get("Detection rate", np.nan)), "Control flags": percent_text(metrics.get("Control flag rate", np.nan)), "Detected / eligible incidents": percent_text(metrics.get("Detected / eligible incident engines", np.nan)), "Unassessed incidents": metrics.get("Unassessed incident engines", "Not established"), "Unassessed controls": metrics.get("Unassessed control engines", "Not established"), "AUC": round(metrics["AUC"], 3) if pd.notna(metrics.get("AUC", np.nan)) else "Not established", "Date/coverage AUC": round(metrics["Date/coverage AUC"], 3) if pd.notna(metrics.get("Date/coverage AUC", np.nan)) else "Not established", "Point target met": "Yes, development estimate only" if metrics.get("Point target met") else "No / not established", "Status": result["status"]}


def p15s_render(state):
    show_message("Focused HPT1 / TRU signal assessment " + P15_SIGNAL_VERSION, "Every confusion count below represents one independent engine assessment. Controls and incidents use the same issue-specific dates and EPS phases. The first documented incident per engine is used; later incidents are not substituted for missing evidence. Results are development cross-validation because the previous run has already informed this revision. Controls are not verified healthy, and recorded EPS generation remains to be confirmed as earlier-only.", "warning")
    if state.get("summary"):
        display_table(pd.DataFrame(state["summary"].values()))
    if state.get("feature_availability"):
        show_message("Measurement support", "EPS reference counts are phase snapshots. Operating-residual features require 100 to 300 reconciled take-off anchors and populated actual measurements. Missing operating-residual features do not establish normal condition.")
        display_table(pd.DataFrame(state["feature_availability"].values()))
    for key, result in state.get("results", {}).items():
        head, lead = key
        metrics = result["metrics"]
        if metrics:
            display_table(pd.DataFrame([{"Issue": head, "Lead days": lead, "Detection 95% range": "{} to {}".format(percent_text(metrics["Detection 95% lower"]), percent_text(metrics["Detection 95% upper"])), "Control-flag 95% range": "{} to {}".format(percent_text(metrics["Control flag 95% lower"]), percent_text(metrics["Control flag 95% upper"])), "Assessment coverage": percent_text(metrics["Assessment coverage"])}]))
        features = result["features"]
        if not features.empty:
            selected = features.groupby("Feature").agg(Selections=("Feature", "size"), Mean_importance=("Importance", "mean")).sort_values(["Selections", "Mean_importance"], ascending=False).head(10).reset_index()
            show_message("{} {} days: repeatedly selected measurements".format(head, lead), "Feature selection is evidence of model use, not proof of component causation. PHYS features are deviations after conditioning on operating inputs; their historical reference is not independently confirmed healthy.")
            display_table(selected)
        repeats = result["repeat_metrics"]
        if len(repeats) > 1:
            shown = repeats[["Repeat", "Detection rate", "Control flag rate", "AUC", "Date/coverage AUC"]].copy()
            for column in ("Detection rate", "Control flag rate"):
                shown[column] = shown[column].map(percent_text)
            display_table(shown)
    unavailable = list(state.get("unavailable", {}).values())
    if unavailable:
        skipped = pd.concat(unavailable, ignore_index=True)
        if not skipped.empty:
            display_table(skipped.groupby(["Issue", "Lead days", "Reason"]).size().reset_index(name="Incident engines"))
    show_message("How to read this result", "A lead period is the time from assessment cutoff to the documented event date, not a measured failure-onset time. AUC describes score ranking; the alert threshold is separately calibrated using controls outside model training. A target met here still requires an untouched test and confirmed healthy controls. Combining HPT1 and TRU alerts requires a separate combined false-alert assessment.", "warning")


def run_focused_signal_analysis(options=None, progress=print):
    required = ("settings", "eps_groups", "eps_columns", "control_engines", "genie_matrix")
    if not isinstance(globals().get("P15"), dict) or any(key not in P15 or P15[key] is None for key in required):
        raise RuntimeError("Run the v6.3 Cells 1 to 4 first in this same Python session; the focused cell reuses their loaded tables")
    if P15.get("runtime_active") or P15.get("signal_active"):
        raise RuntimeError("Finish or cancel the existing analysis before starting another")
    if not str(globals().get("NOTEBOOK_VERSION", "")).startswith("6.3"):
        raise RuntimeError("This focused cell requires the completed v6.3 analysis definitions")
    config = p15s_configuration(options)
    settings = dict(P15["settings"])
    source_key = runtime_key(P15_SIGNAL_VERSION, P15.get("analysis_key"), runtime_scientific_settings(settings), {key: value for key, value in config.items() if key != "resume"}, EVENTS, runtime_frame_signature(P15["genie_matrix"]))
    previous = P15.get("signal_analysis")
    if not config["resume"] or not isinstance(previous, dict) or previous.get("key") != source_key:
        P15["signal_analysis"] = {"key": source_key, "config": config, "source_snapshot": P15.get("analysis_key"), "results": {}, "summary": {}, "unavailable": {}, "status": "Running"}
    state = P15["signal_analysis"]
    P15["signal_active"] = True
    state["status"] = "Running"
    runtime_message(progress, "START {} | reusing loaded cruise/take-off tables; no Genie benchmark rerun".format(P15_SIGNAL_VERSION))
    runtime_message(progress, "One assessment per engine, issue-specific matching, chronological historical references, nested model selection and independent control calibration")
    try:
        for lead in config["lead_days"]:
            for head in config["heads"]:
                result_key = (head, lead)
                if result_key in state["results"]:
                    runtime_message(progress, "REUSE completed focused result: {} {} days".format(head, lead))
                    continue
                with runtime_operation("Issue-specific matching {} {} days".format(head, lead), progress):
                    packet = p15s_matched_engines(head, lead, settings, config, state, progress)
                state["unavailable"][result_key] = packet["unavailable"]
                with runtime_operation("Pre-event signal features {} {} days".format(head, lead), progress):
                    frame, notices = p15s_feature_matrix(packet, settings, config, state, progress)
                state.setdefault("feature_notices", {})[result_key] = notices
                support = {"Issue": head, "Lead days": lead, "Eligible engines": len(frame)}
                for prefix, label in (("LEVEL|", "EPS level/change support"), ("DRIFT|", "Historical EPS drift support"), ("PHYS|", "Operating-residual support")):
                    columns = [column for column in frame if column.startswith(prefix)]
                    support[label] = int(frame[columns].notna().any(axis=1).sum()) if columns else 0
                state.setdefault("feature_availability", {})[result_key] = support
                with runtime_operation("Nested signal assessment {} {} days".format(head, lead), progress):
                    result = p15s_validate(frame, head, lead, config, state, progress)
                state["results"][result_key] = result
                state["summary"][result_key] = p15s_summary_row(head, lead, result, packet)
                display_table(pd.DataFrame(state["summary"].values()))
        state["status"] = "Completed"
        p15s_render(state)
        return pd.DataFrame(state["summary"].values())
    except BaseException as exc:
        state["status"] = "Interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "Failed"
        state["error"] = "{}: {}".format(type(exc).__name__, str(exc))
        runtime_message(progress, "{} focused analysis: {}. Re-run this cell in the same living Python session to reuse completed matching, features and folds.".format(state["status"].upper(), state["error"]))
        raise
    finally:
        P15["signal_active"] = False


P15_SIGNAL_SUMMARY = run_focused_signal_analysis()
