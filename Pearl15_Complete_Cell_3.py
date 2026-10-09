def phase_measurement_choices(frame, family, stage=None):
    subset = (
        frame.loc[frame["__Stage"].eq(stage)]
        if stage and "__Stage" in frame
        else frame
    )
    columns = [
        name
        for name in subset
        if raw_measurement(name) and signal_family(name) == family
    ]
    columns.sort(
        key=lambda name: (
            (
                -float(subset[name].notna().mean())
                if len(subset)
                else 0
            ),
            name,
        ),
    )
    return columns


def raw_event_review(event, settings):
    cutoff = date_boundary(event["date"], settings)
    history = selected_history(event["esn"], cutoff, settings)
    start = history["_flight"].dt.tz_localize(None)
    global_config = {
        "engine": canonical_engine(event["esn"]),
        "event_date": cutoff.tz_localize(None),
        "range_start": None,
        "range_end": None,
        "max_flights": settings["max_flights"],
        "fit_fraction": 0.6,
        "cal_fraction": 0.2,
        "inputs": list(settings["inputs"]),
        "targets": list(settings["targets"]),
        "phases": list(settings["phases"]),
        "coverage": 0.95,
        "persistence": (3, 5),
        "model_mode": "Ridge",
        "epr_confirmed": False,
        "physics_confirmed": False,
        "baseline_confirmed": settings["baseline_confirmed"],
    }
    if not global_config["inputs"] or not global_config["targets"]:
        raise ValueError(
            (
                "No supported operating inputs or meas"
                "ured outputs exist in these catalogue"
                "s."
            ),
        )
    flights = select_flights(start, global_config)
    readings, summaries, unavailable, metrics, checks, models = ([], [], [], [], [], {})
    actual_targets = set()
    for phase in settings["phases"]:
        da = window_rows(
            "".join(("{}".format(phase), " DA")),
            event["esn"],
            history,
            cutoff,
            settings,
        )
        eps = window_rows(
            "".join(("{}".format(phase), " EPS")),
            event["esn"],
            history,
            cutoff,
            settings,
        )
        rows, source_check = merge_measurement_sources(da, eps, settings)
        if rows.empty:
            unavailable.extend(
                (
                    {
                        "Phase": phase,
                        "Parameter": target,
                        "Reason": (
                            "No measurement snapshots "
                            "before the event"
                        ),
                    }
                    for target in settings["targets"]
                ),
            )
            continue
        raw = (
            rows.rename(
                columns={
                    "_flight": "__FlightStart",
                    "_time": "__SnapshotTime",
                },
            )
        ).copy()
        raw["__FlightStart"] = raw["__FlightStart"].dt.tz_localize(None)
        raw["__SnapshotTime"] = raw["__SnapshotTime"].dt.tz_localize(None)
        raw["__Phase"] = phase
        prepared, extraction = prepare_measurements(raw, flights, global_config)
        inputs = []
        for family in ("P20", "T20", "ALT", "NH", "NL", "RPM", "MN"):
            candidates = phase_measurement_choices(
                prepared,
                family,
                "Learning",
            )
            if (
                candidates
                and (
                    (
                        (
                            prepared.loc[prepared["__Stage"].eq(
                                "Learning"), candidates[0]]
                        ).notna()
                    ).mean()
                    >= 0.8
                )
            ):
                if (
                    family != "RPM"
                    or not any(
                        (
                            signal_family(name) in {"NH", "NL"}
                            for name in inputs
                        ),
                    )
                ):
                    inputs.append(candidates[0])
        targets = []
        for family in dict.fromkeys(
            (
                signal_family(target)
                for target in settings["targets"]
            ),
        ):
            candidates = phase_measurement_choices(
                prepared,
                family,
                "Learning",
            )
            if candidates:
                targets.append(candidates[0])
            else:
                placeholder = next(
                    (
                        target
                        for target in settings["targets"]
                        if signal_family(target) == family
                    ),
                )
                unavailable.append(
                    {
                        "Phase": phase,
                        "Parameter": placeholder,
                        "Reason": (
                            "No genuine measurement co"
                            "lumn for this output in t"
                            "his phase"
                        ),
                    },
                )
        actual_targets.update(targets)
        if not inputs or not targets:
            unavailable.extend(
                (
                    {
                        "Phase": phase,
                        "Parameter": target,
                        "Reason": (
                            "No operating inputs have "
                            "adequate learning-period "
                            "coverage"
                        ),
                    }
                    for target in targets
                ),
            )
            continue
        config = {
            **global_config,
            "phases": [phase],
            "inputs": inputs,
            "targets": targets,
        }
        prepared, extraction = prepare_measurements(raw, flights, config)
        bundle = analyse_measurements(prepared, flights, config)
        for destination, key in (
            (readings, "readings"),
            (summaries, "summary"),
            (metrics, "metrics"),
        ):
            if not bundle[key].empty:
                destination.append(bundle[key])
        unavailable.extend(
            bundle["unavailable"].to_dict("records"),
        )
        checks.append(
            {
                "Phase": phase,
                "Inputs": ", ".join(inputs),
                **source_check,
                **extraction,
            },
        )
        models.update(bundle["models"])
    config = {
        **global_config,
        "targets": sorted(actual_targets | set(settings["targets"])),
    }
    result = {
        "config": config,
        "flights": flights,
        "models": models,
        "readings": (
            pd.concat(readings, ignore_index=True)
            if readings
            else pd.DataFrame()
        ),
        "summary": (
            pd.concat(summaries, ignore_index=True)
            if summaries
            else pd.DataFrame()
        ),
        "metrics": (
            pd.concat(metrics, ignore_index=True)
            if metrics
            else pd.DataFrame()
        ),
        "unavailable": pd.DataFrame(unavailable),
        "extraction_checks": pd.DataFrame(checks),
        "event": event,
        "identity": "; ".join(history["_identity"].drop_duplicates()),
    }
    P15["raw_reviews"][event["id"]] = result
    return result


def eps_event_review(event, settings):
    cutoff = date_boundary(event["date"], settings)
    history = selected_history(event["esn"], cutoff, settings)
    n = len(history)
    learning_end, reference_end = (int(n * 0.6), int(n * 0.8))
    monitored_cycles = history["_cycle"].iloc[reference_end:]
    family_records, point_frames, rejected = ([], [], [])
    for phase in settings["phases"]:
        snapshots = window_rows(
            "".join(("{}".format(phase), " EPS")),
            event["esn"],
            history,
            cutoff,
            settings,
        )
        source = P15["data"].get(
            "".join(("{}".format(phase), " EPS")),
            pd.DataFrame(),
        )
        signals = [
            name
            for name in source.attrs.get("signals", [])
            if name in snapshots
        ]
        if not signals or snapshots.empty:
            rejected.append(
                {
                    "Phase": phase,
                    "Reason": (
                        "No usable EPS health channels"
                        " before the event"
                    ),
                },
            )
            continue
        cycles = snapshots.groupby("_cycle")[signals].median().reindex(
            history["_cycle"],
        )
        learning, reference = (
            cycles.iloc[:learning_end],
            cycles.iloc[learning_end:reference_end],
        )
        center = learning.median()
        scale = learning.sub(center).abs().median() * 1.4826
        usable = (
            (
                learning.notna().sum().ge(20)
                & reference.notna().sum().ge(8)
            )
            & scale.gt(1e-09)
        )
        for family in sorted(
            (
                {signal_family(signal) for signal in signals}
                & (OUTPUT_FAMILIES | {"EPR"})
            ),
        ):
            columns = [
                name
                for name in signals
                if (
                    signal_family(name) == family
                    and usable.get(name, False)
                )
            ]
            if not columns:
                rejected.append(
                    {
                        "Phase": phase,
                        "Parameter family": family,
                        "Reason": (
                            "EPS channels lack suffici"
                            "ent varying learning/refe"
                            "rence readings for a stab"
                            "le deviation band"
                        ),
                    },
                )
                continue
            z = cycles[columns].sub(center[columns]).div(
                scale[columns],
            )
            expected_count = len(columns)
            available = z.notna().sum(axis=1).ge(
                max(1, int(np.ceil(expected_count * 0.5))),
            )
            severity = z.abs().max(axis=1).where(available)
            reference_scores = (
                severity.iloc[learning_end:reference_end].dropna()
            ).sort_values()
            if len(reference_scores) < 8:
                rejected.append(
                    {
                        "Phase": phase,
                        "Parameter family": family,
                        "Reason": (
                            "Fewer than 8 comparable E"
                            "PS reference flights"
                        ),
                    },
                )
                continue
            rank = min(
                len(reference_scores),
                int(np.ceil((len(reference_scores) + 1) * 0.95)),
            )
            threshold = max(float(reference_scores.iloc[rank - 1]), 1e-09)
            normalized = severity / threshold
            monitoring = normalized.loc[monitored_cycles]
            flags = (
                monitoring.gt(1).where(
                    monitoring.notna(),
                    np.nan,
                )
            ).astype(
                float,
            )
            persistent = persistent_flags(flags, 3, 5)
            first = (
                persistent.index[persistent].min()
                if persistent.any()
                else None
            )
            reference_bias = z.iloc[learning_end:reference_end].median().abs().max()
            stable = bool(reference_bias <= 3.0)
            assessed = int(monitoring.notna().sum())
            first_time = (
                history.set_index("_cycle").loc[first, "_flight"]
                if first is not None
                else pd.NaT
            )
            driver = (
                z.loc[first].abs().idxmax()
                if first is not None
                else columns[0]
            )
            direction = (
                "Above earlier baseline"
                if first is not None and z.loc[first, driver] > 0
                else (
                    "Below earlier baseline"
                    if first is not None
                    else "No persistent deviation"
                )
            )
            family_records.append(
                {
                    "Phase": phase,
                    "Parameter family": family,
                    "EPS channels": len(columns),
                    "Reference flights": len(reference_scores),
                    "Reference meaning": (
                        "Empirical family band; no gua"
                        "ranteed false-alarm probabili"
                        "ty"
                    ),
                    "Assessed monitoring flights": assessed,
                    "Unassessed monitoring flights": len(monitored_cycles) - assessed,
                    "Deviation flight share": (
                        float(flags.eq(1).sum() / assessed)
                        if assessed
                        else np.nan
                    ),
                    "Persistent deviation flight share": (
                        float(persistent.sum() / assessed)
                        if assessed
                        else np.nan
                    ),
                    "First persistent deviation": first_time,
                    "Lead days": (
                        (
                            (cutoff - first_time).total_seconds()
                            / 86400
                        )
                        if pd.notna(first_time)
                        else np.nan
                    ),
                    "Leading EPS channel": driver,
                    "Direction": direction,
                    "Result": (
                        "Persistent EPS parameter deviation"
                        if first is not None and stable
                        else (
                            "Baseline shift requires review"
                            if not stable
                            else (
                                (
                                    "No persistent dev"
                                    "iation in assesse"
                                    "d flights"
                                )
                                if assessed
                                else (
                                    "No assessable mon"
                                    "itoring flights"
                                )
                            )
                        )
                    ),
                    "Baseline": (
                        (
                            "Earlier baseline confirme"
                            "d against records"
                        )
                        if settings["baseline_confirmed"]
                        else (
                            "Earlier baseline assumed "
                            "healthy; unverified"
                        )
                    ),
                },
            )
            points = history[["_flight", "_cycle"]].copy()
            points["Phase"], points["Parameter family"] = (phase, family)
            points["Severity / reference band"] = normalized.to_numpy()
            points["Stage"] = (
                (
                    ["Learning"] * learning_end
                    + (
                        ["Reference"]
                        * (reference_end - learning_end)
                    )
                )
                + ["Monitoring"] * (n - reference_end)
            )
            points["Persistent deviation"] = (
                points["_cycle"].isin(
                    persistent.index[persistent],
                )
                & stable
            )
            point_frames.append(points)
    result = {
        "event": event,
        "history": history,
        "summary": pd.DataFrame(family_records),
        "points": (
            pd.concat(point_frames, ignore_index=True)
            if point_frames
            else pd.DataFrame()
        ),
        "unavailable": pd.DataFrame(rejected),
    }
    P15["eps_reviews"][event["id"]] = result
    return result


def review_all_events(settings, progress=print):
    rows, errors = ([], [])
    for position, event in enumerate(
        sorted(
            EVENTS,
            key=lambda event: (event_category(event), event["date"], event["id"]),
        ),
    ):
        eps, raw = (None, None)
        for name, function in (
            ("EPS parameter review", eps_event_review),
            ("Raw measurement review", raw_event_review),
        ):
            try:
                result = function(event, settings)
                if name.startswith("EPS"):
                    eps = result
                else:
                    raw = result
            except (ValueError, KeyError) as exc:
                errors.append(
                    {
                        "Event": event["id"],
                        "Engine": event["esn"],
                        "Issue": event_category(event),
                        "Review": name,
                        "Reason": str(exc),
                    },
                )
        head = P15["validation"].get(event_category(event), {})
        prediction = head.get("test", pd.DataFrame())
        prediction = (
            prediction.loc[prediction["sample_id"].eq(event["id"])]
            if not prediction.empty
            else prediction
        )
        row = {
            "Event": event["id"],
            "Engine": event["esn"],
            "Issue": event_category(event),
            "Event date": event["date"],
            "EPS review": (
                "Available"
                if eps and (not eps["summary"].empty)
                else "Unavailable"
            ),
            "Raw models fitted": len(raw["models"]) if raw else 0,
            "Classifier result": "Not established",
            "Classifier score": np.nan,
            "Classifier reason": (
                "; ".join(head.get("warnings", []))
                if prediction.empty
                else prediction.iloc[0]["reason"]
            ),
        }
        if not prediction.empty:
            scored = prediction.iloc[0]
            row["Classifier score"] = scored["score"]
            row["Classifier result"] = (
                "Retrospective candidate flag"
                if scored["alert_assessed"] and scored["flag"]
                else (
                    "No flag at assessed cutoff"
                    if scored["alert_assessed"]
                    else (
                        (
                            "Score available; threshol"
                            "d not established"
                        )
                        if scored["assessed"]
                        else "Not established"
                    )
                )
            )
        if eps and (not eps["summary"].empty):
            candidates = eps["summary"].loc[eps["summary"]["Result"].eq(
                "Persistent EPS parameter deviation",
            )]
            row["EPS parameter deviations"] = (
                "; ".join(
                    (
                        candidates["Phase"] + " "
                        + candidates["Parameter family"]
                    ),
                )
                if not candidates.empty
                else "None accepted in assessed flights"
            )
            row["First EPS deviation"] = (
                candidates["First persistent deviation"].min()
                if not candidates.empty
                else pd.NaT
            )
        else:
            (
                row["EPS parameter deviations"],
                row["First EPS deviation"],
            ) = ("Not established", pd.NaT)
        rows.append(row)
        if (position + 1) % 10 == 0:
            progress(
                "".join(
                    (
                        "Reviewed ",
                        "{}".format(position + 1),
                        "/",
                        "{}".format(len(EVENTS)),
                        " documented events",
                    ),
                ),
            )
    P15["incident_report"], P15["case_errors"] = (
        pd.DataFrame(rows), pd.DataFrame(errors))
    return P15["incident_report"]


def show_message(title, text, level="info"):
    colors = {
        "info": ("#eff6ff", "#1e40af"),
        "warning": ("#fffbeb", "#92400e"),
        "error": ("#fef2f2", "#991b1b"),
        "success": ("#ecfdf5", "#065f46"),
    }
    background, color = colors[level]
    display(
        HTML(
            "".join(
                (
                    (
                        "<div style='padding:16px;bord"
                        "er-radius:8px;background:"
                    ),
                    "{}".format(background),
                    ";color:",
                    "{}".format(color),
                    (
                        ";margin:10px 0'><strong style"
                        "='font-size:18px'>"
                    ),
                    "{}".format(html.escape(title)),
                    "</strong><div style='margin-top:7px'>",
                    "{}".format(html.escape(text)),
                    "</div></div>",
                ),
            ),
        ),
    )


def percent_text(value):
    return (
        "{:.1%}".format(value)
        if pd.notna(value) and np.isfinite(value)
        else "Not established"
    )


def ratio_text(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return "Not established"
    return "".join(
        (
            "{}".format(int(numerator)),
            " / ",
            "{}".format(int(denominator)),
        ),
    )


def display_table(frame, missing=None):
    replacements = {
        "First persistent warning": "No accepted warning recorded",
        "Warning snapshot time": "No accepted warning recorded",
        "First unusual flight": "No unusual reading recorded",
        "First persistent deviation": "No persistent deviation recorded",
        "First EPS deviation": "No accepted deviation recorded",
        "Lead days": "Not applicable without a deviation",
        "Days before event date": "Not applicable without a warning",
        "Recorded flights before event": "Not applicable without a warning",
        "Classifier score": "Not established",
    }
    replacements.update(missing or {})
    shown = frame.astype(object).copy()
    for column in shown:
        shown[column] = shown[column].where(
            shown[column].notna(),
            replacements.get(column, "Not established"),
        )
    display(shown)
    return shown


def issue_summary(head):
    result = P15["validation"][head]
    metrics = result.get("metrics", {})
    documented = (
        EVENTS
        if head == "ANY"
        else [
            event
            for event in EVENTS
            if event_category(event) == head
        ]
    )
    return {
        "Issue": head,
        "Documented events": len(documented),
        "Documented engines": len({event["esn"] for event in documented}),
        "Detection": percent_text(metrics.get("Detection rate", np.nan)),
        "Detected / assessed incidents": ratio_text(
            metrics.get("Detected incident windows", np.nan),
            metrics.get("Test incident windows", 0),
        ),
        "Control-engine flags": percent_text(metrics.get("Control flag rate", np.nan)),
        "Flagged / assessed controls": ratio_text(
            metrics.get("Flagged control engines", np.nan),
            metrics.get("Test control engines", 0),
        ),
        "Assessment coverage": percent_text(metrics.get("Assessment coverage", np.nan)),
        "AUC": (
            round(metrics["AUC"], 3)
            if pd.notna(metrics.get("AUC", np.nan))
            else "Not established"
        ),
        "Date/coverage-only AUC": (
            round(metrics["Date/coverage-only AUC"], 3)
            if pd.notna(
                metrics.get("Date/coverage-only AUC", np.nan),
            )
            else "Not established"
        ),
        "Result": result["status"],
    }


def render_fleet_summary():
    show_message(
        "Pearl-15 automatic review",
        (
            "EPS classifiers follow a retrospective en"
            "gine-held-out approach. Each incident use"
            "s only its pre-event snapshots; controls "
            "use matching assessment dates. The date/h"
            "istory/coverage audit completes the check"
            " Genie left unfinished. Percentages below"
            " are measured locally after this run, not"
            " copied from Genie."
        ),
    )
    summary = pd.DataFrame(
        [
            issue_summary(head)
            for head in ("HPT1", "HPT2", "TRU", "ANY")
        ],
    )
    display_table(summary)
    for head, result in P15["validation"].items():
        metrics = result.get("metrics", {})
        show_message(
            "".join(
                (
                    "{}".format(head),
                    ": ",
                    "{}".format(result["status"]),
                ),
            ),
            " ".join(result.get("warnings", [])),
            "warning",
        )
        if metrics:
            details = pd.DataFrame(
                [
                    {
                        "Measure": "Detection 95% range",
                        "Result": "".join(
                            (
                                "{}".format(
                                    percent_text(
                                        metrics.get(
                                            "Detection 95% lower",
                                            np.nan,
                                        ),
                                    ),
                                ),
                                " to ",
                                "{}".format(
                                    percent_text(
                                        metrics.get(
                                            "Detection 95% upper",
                                            np.nan,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    },
                    {
                        "Measure": "Control-flag 95% range",
                        "Result": "".join(
                            (
                                "{}".format(
                                    percent_text(
                                        metrics.get(
                                            "Control flag 95% lower",
                                            np.nan,
                                        ),
                                    ),
                                ),
                                " to ",
                                "{}".format(
                                    percent_text(
                                        metrics.get(
                                            "Control flag 95% upper",
                                            np.nan,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    },
                    {
                        "Measure": "Requested targets",
                        "Result": "".join(
                            (
                                "Detection >= ",
                                "{:.0%}".format(
                                    P15["settings"]["detection_target"],
                                ),
                                "; control-engine flags <= ",
                                "{:.0%}".format(
                                    P15["settings"]["false_alarm_limit"],
                                ),
                            ),
                        ),
                    },
                    {
                        "Measure": "Measured point targets",
                        "Result": (
                            "Met in primary retrospective repeat"
                            if metrics.get(
                                "Point target met",
                                False,
                            )
                            else "Not met or not established"
                        ),
                    },
                ],
            )
            display_table(details)
        repeats = result.get("repeat_metrics")
        if isinstance(repeats, pd.DataFrame) and (not repeats.empty):
            repeated = (
                repeats.reindex(
                    columns=[
                        "Repeat",
                        "Detection rate",
                        "Control flag rate",
                        "AUC",
                        "Date/coverage-only AUC",
                        "Unassessed test windows",
                    ],
                )
            ).copy()
            for column in ("Detection rate", "Control flag rate"):
                repeated[column] = repeated[column].map(percent_text)
            display_table(repeated)
        elif metrics:
            show_message(
                "".join(
                    (
                        "{}".format(head),
                        ": repeat stability unavailable",
                    ),
                ),
                (
                    "No repeat-metrics table is presen"
                    "t for this result. Available prim"
                    "ary metrics remain shown; repeat "
                    "stability is not established."
                ),
                "warning",
            )
        download_csv(
            result.get("test", pd.DataFrame()),
            "".join(
                (
                    "Pearl15_",
                    "{}".format(head),
                    "_heldout_windows.csv",
                ),
            ),
        )
        download_csv(
            result.get("folds", pd.DataFrame()),
            "".join(
                (
                    "Pearl15_",
                    "{}".format(head),
                    "_fold_checks.csv",
                ),
            ),
        )
    if not P15["cohort_skipped"].empty:
        compact = (
            (
                P15["cohort_skipped"].groupby(
                    ["Category", "Reason"],
                )
            ).size()
        ).reset_index(
            name="Skipped records",
        )
        show_message(
            "Classifier eligibility checks",
            (
                "Unavailable classifier windows remain"
                " eligible for individual parameter re"
                "view when their measurements support "
                "it."
            ),
            "warning",
        )
        display_table(compact)
    if not P15["case_errors"].empty:
        compact = (
            P15["case_errors"].groupby(["Review", "Reason"]).size()
        ).reset_index(
            name="Affected events",
        )
        show_message(
            "Parameter review availability",
            (
                "These are actual data or model-refere"
                "nce limitations. Missing evidence doe"
                "s not establish a healthy engine."
            ),
            "warning",
        )
        display_table(compact)
    download_csv(summary, "Pearl15_issue_percentages.csv")
    download_csv(
        P15["incident_report"],
        "Pearl15_all_incident_results.csv",
    )
    download_csv(
        P15["case_errors"],
        "Pearl15_unavailable_reviews.csv",
    )
    download_json(P15["manifest"], "Pearl15_run_manifest.json")


def draw_eps_incident(bundle):
    points = bundle["points"]
    if points.empty:
        return None
    event = bundle["event"]
    figure, axes = plt.subplots(
        len(P15["settings"]["phases"]),
        1,
        figsize=(13, 3.6 * len(P15["settings"]["phases"])),
        squeeze=False,
        dpi=100,
    )
    cutoff = date_boundary(event["date"], P15["settings"])
    for axis, phase in zip(axes.flat, P15["settings"]["phases"]):
        phase_rows = points.loc[points["Phase"].eq(phase)]
        if phase_rows.empty:
            axis.text(
                0.5,
                0.5,
                (
                    "No EPS channels with a stable pre"
                    "-event reference"
                ),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue
        for family, rows in phase_rows.groupby("Parameter family"):
            axis.plot(
                rows["_flight"],
                rows["Severity / reference band"],
                label=family,
                linewidth=1,
            )
            warning = rows.loc[rows["Persistent deviation"]]
            if not warning.empty:
                axis.scatter(
                    warning["_flight"],
                    warning["Severity / reference band"],
                    s=15,
                    zorder=4,
                )
        for stage, color in (
            ("Learning", "#f1f5f9"),
            ("Reference", "#eff6ff"),
            ("Monitoring", "#ffffff"),
        ):
            times = phase_rows.loc[phase_rows["Stage"].eq(stage), "_flight"]
            if not times.empty:
                axis.axvspan(
                    times.min(),
                    times.max(),
                    color=color,
                    alpha=0.5,
                )
        axis.axhline(
            1,
            color="#dc2626",
            linestyle="--",
            linewidth=1,
        )
        axis.axvline(cutoff, color="#334155", linestyle=":")
        axis.set(
            title=phase,
            ylabel="EPS deviation / reference band",
            xlabel="UTC date; event day excluded",
        )
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8, ncol=5)
    figure.suptitle(
        "".join(
            (
                "{}".format(event_category(event)),
                " \u00b7 ESN ",
                "{}".format(event["esn"]),
                " \u00b7 ",
                "{}".format(event["date"]),
                (
                    "\nParameter evidence; earlier hea"
                    "lthy baseline remains an assumpti"
                    "on unless confirmed"
                ),
            ),
        ),
        fontsize=12,
    )
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def render_issue_cases(category, report_settings):
    selected = [
        event
        for event in EVENTS
        if (
            event_category(event) == category
            and (
                not report_settings["engines"]
                or (
                    canonical_engine(event["esn"])
                    in report_settings["engines"]
                )
            )
        )
    ]
    selected.sort(key=lambda event: (event["date"], event["id"]))
    if report_settings["case_limit"] is not None:
        selected = selected[:report_settings["case_limit"]]
    show_message(
        "".join(
            (
                "{}".format(category),
                ": ",
                "{}".format(len(selected)),
                " incident reports",
            ),
        ),
        (
            "Per-parameter percentages are the share o"
            "f assessable monitoring flight anchors wi"
            "th deviations. They are not failure proba"
            "bilities. Classifier evaluation and param"
            "eter evidence answer different questions."
        ),
    )
    report = P15.get("incident_report", pd.DataFrame())
    if not report.empty:
        display_table(
            (
                report.loc[report["Event"].isin(
                    [event["id"] for event in selected],
                )]
            ).drop(
                columns=["Classifier reason"],
            ),
        )
    for event in selected:
        show_message(
            "".join(
                (
                    "ESN ",
                    "{}".format(event["esn"]),
                    " \u00b7 ",
                    "{}".format(event["date"]),
                ),
            ),
            event["kind"],
        )
        eps = P15["eps_reviews"].get(event["id"])
        if eps:
            history = eps["history"]
            gap = (
                (
                    (
                        date_boundary(
                            event["date"],
                            P15["settings"],
                        )
                        - history["_flight"].iloc[-1]
                    )
                ).total_seconds()
                / 86400
            )
            show_message(
                "History used",
                (
                    "".join(
                        (
                            "{}".format(len(history)),
                            (
                                " reconciled flight an"
                                "chors; last available"
                                " take-off "
                            ),
                            "{:.1f}".format(gap),
                            " days before the event. ",
                        ),
                    )
                    + "; ".join(
                        history["_identity"].drop_duplicates(),
                    )
                ),
                (
                    "warning"
                    if gap > P15["settings"]["max_gap_days"]
                    else "info"
                ),
            )
            summary = eps["summary"].copy()
            for column in (
                "Deviation flight share",
                "Persistent deviation flight share",
            ):
                if column in summary:
                    summary[column] = summary[column].map(percent_text)
            display_table(summary)
            figure = draw_eps_incident(eps)
            if figure is not None:
                display(figure)
                plt.close(figure)
            download_csv(
                eps["summary"],
                "".join(
                    (
                        "Pearl15_",
                        "{}".format(event["id"]),
                        "_EPS_parameters.csv",
                    ),
                ),
            )
        model = P15["event_models"].get(category, {}).get(event["id"])
        cohort = P15["cohort"]
        window = (
            cohort.loc[cohort["sample_id"].eq(event["id"])]
            if not cohort.empty
            else cohort
        )
        held = P15["validation"].get(category, {}).get(
            "test",
            pd.DataFrame(),
        )
        held = (
            held.loc[held["sample_id"].eq(event["id"])]
            if not held.empty
            else held
        )
        assessed_score = bool(
            (
                not held.empty
                and held.iloc[0]["assessed"]
                and pd.notna(held.iloc[0]["score"])
            ),
        )
        if (
            model is not None
            and (not window.empty)
            and assessed_score
        ):
            evidence = parameter_evidence(model, window.iloc[0].to_dict())
            show_message(
                (
                    "Parameters influencing the held-o"
                    "ut classifier score"
                ),
                (
                    "This ranking comes from a model t"
                    "rained and calibrated on other en"
                    "gines. It shows model sensitivity"
                    ", not confirmed physical causatio"
                    "n."
                ),
            )
            display_table(evidence.head(8))
        raw = P15["raw_reviews"].get(event["id"])
        if raw:
            checks = raw.get("extraction_checks", pd.DataFrame())
            if not checks.empty:
                if (
                    (
                        checks.get(
                            "Raw DA/EPS conflict cells excluded",
                            pd.Series(dtype=int),
                        )
                    ).gt(
                        0,
                    )
                ).any():
                    show_message(
                        "Conflicting DA/EPS measurements",
                        (
                            "Some shared raw measureme"
                            "nts disagree at the same "
                            "engine/snapshot timestamp"
                            ". Those individual values"
                            " are excluded from the pa"
                            "rameter model; EPS health"
                            " channels remain availabl"
                            "e independently."
                        ),
                        "warning",
                    )
                display_table(checks)
            percentages = raw_output_percentages(raw)
            render_output_percentage_table(percentages)
            if not raw["unavailable"].empty:
                display_table(raw["unavailable"])
            for phase, target in raw["models"]:
                if (
                    signal_family(target)
                    in report_settings["graph_outputs"]
                ):
                    figure = draw_raw_continuous(
                        raw,
                        phase,
                        target,
                        calendar=True,
                    )
                    display(figure)
                    plt.close(figure)
            download_csv(
                percentages,
                "".join(
                    (
                        "Pearl15_",
                        "{}".format(event["id"]),
                        "_raw_parameter_percentages.csv",
                    ),
                ),
            )
        if not eps and (not raw):
            errors = P15["case_errors"]
            display_table(
                errors.loc[errors["Event"].eq(event["id"])],
            )


def raw_output_percentages(bundle):
    config = bundle["config"]
    readings = bundle["readings"]
    total = int(bundle["flights"]["__Stage"].eq("Monitoring").sum())
    records = []
    for phase in config["phases"]:
        targets = {
            target
            for candidate_phase, target in bundle["models"]
            if candidate_phase == phase
        }
        if not bundle["unavailable"].empty:
            targets.update(
                bundle["unavailable"].loc[bundle["unavailable"]
                                          ["Phase"].eq(phase), "Parameter"],
            )
        for target in sorted(targets):
            row = {
                "Phase": phase,
                "Parameter": target,
                "Monitoring flight anchors": total,
                "Assessable flight anchors": 0,
                "Unassessed flight anchors": total,
                "Assessment coverage": "0.0%" if total else "Not established",
                "Deviation flights / assessable": "Not established",
                "Deviation flight share": "Not established",
                "Persistent warning flights / assessable": "Not established",
                "Persistent warning flight share": "Not established",
                "Status": "Unassessed",
                "Reason": "No fitted readings for this phase/output",
            }
            model = bundle["models"].get((phase, target))
            if model is not None and (not readings.empty):
                selected = readings.loc[(
                    (
                        readings["__Phase"].eq(phase)
                        & readings["Parameter"].eq(target)
                    )
                    & readings["__Stage"].eq("Monitoring")
                )]
                comparable = (
                    (
                        selected["In learned conditions"]
                        & selected["Prediction physically admissible"]
                    )
                    & np.isfinite(selected["Score"])
                )
                assessed = selected.loc[comparable]
                count = assessed["__WindowCycle"].nunique()
                unusual = (
                    assessed.loc[assessed["Unusual reading"], "__WindowCycle"]
                ).nunique()
                persistent = (
                    assessed.loc[assessed["Persistent warning"], "__WindowCycle"]
                ).nunique()
                summary = bundle["summary"]
                matching = (
                    summary.loc[(
                        summary["Phase"].eq(phase)
                        & summary["Parameter"].eq(target)
                    )]
                    if not summary.empty
                    else summary
                )
                row.update(
                    {
                        "Assessable flight anchors": count,
                        "Unassessed flight anchors": total - count,
                        "Assessment coverage": percent_text(
                            count / total if total else np.nan,
                        ),
                        "Deviation flights / assessable": ratio_text(unusual, count),
                        "Deviation flight share": percent_text(
                            unusual / count if count else np.nan,
                        ),
                        "Status": (
                            matching.iloc[0]["Status"]
                            if not matching.empty
                            else "Review fitted model"
                        ),
                        "Reason": "; ".join(
                            model.get("review_reasons", []),
                        ),
                    },
                )
                if model["baseline_accepted"]:
                    row["Persistent warning flights / assessable"] = ratio_text(
                        persistent, count)
                    row["Persistent warning flight share"] = percent_text(
                        persistent / count if count else np.nan,
                    )
                if not count:
                    row["Status"] = "No assessable monitoring flights"
                    row["Reason"] = "; ".join(
                        filter(
                            None,
                            [
                                row["Reason"],
                                (
                                    "Monitoring condit"
                                    "ions or predictio"
                                    "ns fall outside t"
                                    "he model's admiss"
                                    "ible reference; n"
                                    "o valid deviation"
                                    " percentage is av"
                                    "ailable."
                                ),
                            ],
                        ),
                    )
            else:
                unavailable = bundle["unavailable"]
                if not unavailable.empty:
                    failures = unavailable.loc[(
                        unavailable["Phase"].eq(phase)
                        & unavailable["Parameter"].eq(target)
                    )]
                    if not failures.empty:
                        row["Reason"] = failures.iloc[0]["Reason"]
            records.append(row)
    return pd.DataFrame(records)


def render_output_percentage_table(report):
    rows = []
    for source in report.to_dict("records"):
        row = {
            key: source[key]
            for key in (
                "Event",
                "Engine",
                "Report date",
                "Phase",
                "Parameter",
            )
            if key in source
        }
        row["Assessment coverage"] = source.get(
            "Assessment coverage", "Not established")
        if (
            "Assessable flight anchors" in source
            and pd.notna(source["Assessable flight anchors"])
        ):
            row["Assessment coverage"] += "".join(
                (
                    " (",
                    "{}".format(
                        int(source["Assessable flight anchors"]),
                    ),
                    "/",
                    "{}".format(
                        int(source["Monitoring flight anchors"]),
                    ),
                    ")",
                ),
            )
        for label, percentage, counts in [
            (
                "Deviation flights",
                "Deviation flight share",
                "Deviation flights / assessable",
            ),
            (
                "Persistent warning flights",
                "Persistent warning flight share",
                "Persistent warning flights / assessable",
            ),
        ]:
            row[label] = source.get(percentage, "Not established")
            if (
                (
                    source.get(counts, "Not established")
                    != "Not established"
                )
                and pd.notna(source.get(counts))
            ):
                row[label] += " (" + source[counts] + ")"
        row["Result"] = source["Status"]
        row["Reason"] = source.get("Reason", "")
        rows.append(row)
    display_table(pd.DataFrame(rows))


def download_csv(frame, name):
    if frame is None or frame.empty:
        return
    payload = base64.b64encode(frame.to_csv(index=False).encode()).decode()
    display(
        HTML(
            "".join(
                (
                    "<a download='",
                    "{}".format(html.escape(name, quote=True)),
                    "' href='data:text/csv;base64,",
                    "{}".format(payload),
                    (
                        "' style='display:inline-block"
                        ";padding:8px 12px;background:"
                        "#f1f5f9;border-radius:6px;mar"
                        "gin:5px'>"
                    ),
                    "{}".format(html.escape(name)),
                    "</a>",
                ),
            ),
        ),
    )


def download_json(value, name):
    payload = (
        base64.b64encode(
            json.dumps(value, indent=2, default=str).encode(),
        )
    ).decode()
    display(
        HTML(
            "".join(
                (
                    "<a download='",
                    "{}".format(html.escape(name, quote=True)),
                    "' href='data:application/json;base64,",
                    "{}".format(payload),
                    (
                        "' style='display:inline-block"
                        ";padding:8px 12px;background:"
                        "#f1f5f9;border-radius:6px;mar"
                        "gin:5px'>"
                    ),
                    "{}".format(html.escape(name)),
                    "</a>",
                ),
            ),
        ),
    )


def draw_raw_continuous(bundle, phase, target, calendar=False):
    data = bundle["readings"]
    if data.empty:
        raise ValueError(
            "No fitted raw parameter readings are available.",
        )
    selected = (
        data.loc[data["__Phase"].eq(phase) & data["Parameter"].eq(target)]
    ).copy()
    if selected.empty:
        raise ValueError(
            (
                "This phase/output could not be fitted"
                "; see the unavailable-parameter table"
                "."
            ),
        )
    valid = (
        selected["In learned conditions"]
        & selected["Prediction physically admissible"]
    )
    selected.loc[~valid, ["Expected", "Lower", "Upper", "Score"]] = np.nan
    grouped = selected.groupby("__WindowCycle").agg(
        Observed=("Observed", "median"),
        Expected=("Expected", "median"),
        Lower=("Lower", "median"),
        Upper=("Upper", "median"),
        Score=("Score", "median"),
        ScoreMin=("Score", "min"),
        ScoreMax=("Score", "max"),
    )
    flights = bundle["flights"].set_index("__WindowCycle")
    plot = flights.join(grouped)
    x = plot["__FlightStart"] if calendar else plot["__PlotCycle"]
    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(13, 7),
        dpi=110,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].plot(
        x,
        plot["Observed"],
        color="#334155",
        label="Measured flight median",
        linewidth=1.6,
    )
    axes[0].plot(
        x,
        plot["Expected"],
        color="#2563eb",
        label="Expected flight median",
        linewidth=1.4,
    )
    axes[0].fill_between(
        x,
        plot["Lower"],
        plot["Upper"],
        color="#cbd5e1",
        alpha=0.55,
        label="Empirical reference band",
    )
    axes[1].plot(
        x,
        plot["Score"],
        color="#2563eb",
        linewidth=1.4,
        label="Median residual / band",
    )
    axes[1].fill_between(
        x,
        plot["ScoreMin"],
        plot["ScoreMax"],
        color="#cbd5e1",
        alpha=0.45,
        label="Snapshot score range",
    )
    for stage in ("Learning", "Reference", "Monitoring"):
        positions = np.flatnonzero(plot["__Stage"].eq(stage).to_numpy())
        if len(positions):
            left = x.iloc[positions[0]]
            next_position = positions[-1] + 1
            right = (
                x.iloc[next_position]
                if next_position < len(x)
                else (
                    pd.Timestamp(bundle["config"]["event_date"])
                    if calendar
                    else 0
                )
            )
            shade = {
                "Learning": "#f1f5f9",
                "Reference": "#eff6ff",
                "Monitoring": "#ffffff",
            }[stage]
            for axis in axes:
                axis.axvspan(
                    left,
                    right,
                    facecolor=shade,
                    alpha=0.45,
                    zorder=0,
                )
            axes[0].text(
                left,
                1.01,
                stage,
                transform=axes[0].get_xaxis_transform(),
                color="#475569",
                fontsize=9,
            )
    for bound in (-1, 1):
        axes[1].axhline(
            bound,
            color="#dc2626",
            linestyle="--",
            linewidth=0.8,
        )
    warnings_frame = selected.loc[selected["Persistent warning"]]
    if not warnings_frame.empty:
        warning_x = (
            warnings_frame["__SnapshotTime"]
            if calendar
            else warnings_frame["__PlotCycle"]
        )
        axes[0].scatter(
            warning_x,
            warnings_frame["Observed"],
            color="#dc2626",
            marker="D",
            s=26,
            label="Persistent parameter deviation",
            zorder=5,
        )
        axes[1].scatter(
            warning_x,
            warnings_frame["Score"],
            color="#dc2626",
            s=22,
            zorder=5,
        )
    event_x = (
        pd.Timestamp(bundle["config"]["event_date"])
        if calendar
        else 0
    )
    for axis in axes:
        axis.axvline(
            event_x,
            color="#0f172a",
            linestyle=":",
            linewidth=1,
        )
        axis.grid(alpha=0.15)
    axes[0].set_ylabel(
        "".join(
            (
                "{}".format(target),
                " (",
                "{}".format(measurement_unit(target)),
                ")",
            ),
        ),
    )
    axes[1].set_ylabel("Residual / band")
    axes[1].set_xlabel(
        (
            "UTC date; event day excluded"
            if calendar
            else (
                "Recorded flight anchors before the ev"
                "ent (event = 0)"
            )
        ),
    )
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8)
    model = bundle["models"][phase, target]
    verdict = (
        "Baseline/model requires review"
        if not model["baseline_accepted"]
        else "Parameter deviation review"
    )
    figure.suptitle(
        "".join(
            (
                "ESN ",
                "{}".format(bundle["event"]["esn"]),
                " \u00b7 ",
                "{}".format(phase),
                " \u00b7 ",
                "{}".format(target),
                "\n",
                "{}".format(verdict),
                " \u00b7 ",
                "{}".format(bundle["identity"]),
            ),
        ),
        fontsize=12,
        fontweight="bold",
    )
    if calendar:
        figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def render_genie_reconstruction():
    frame, keep = (P15["genie_matrix"], P15["genie_keep"])
    show_message(
        "Reconstruction from supplied Genie code",
        (
            "The source uses one row per documented ev"
            "ent and one per control engine. Its "
            "\u201c320 engines\u201d label describes 3"
            "20 rows; repeated events can belong to th"
            "e same engine. Recent levels and changes "
            "are both reconstructed. Live catalogue co"
            "unts may differ from the saved experiment"
            "."
        ),
    )
    counts = pd.DataFrame(
        [
            {
                "Measure": "Assessment rows",
                "Source reference": 320,
                "This run": len(frame),
            },
            {
                "Measure": "Unique engines",
                "Source reference": "Not specified as a distinct count",
                "This run": frame["engine"].nunique(),
            },
            {
                "Measure": "Documented event rows",
                "Source reference": 81,
                "This run": int(frame["category"].ne("Control").sum()),
            },
            {
                "Measure": "Control engines",
                "Source reference": 239,
                "This run": int(frame["category"].eq("Control").sum()),
            },
            {
                "Measure": "Globally retained features",
                "Source reference": 501,
                "This run": len(keep),
            },
            {
                "Measure": "Recent-level features",
                "Source reference": 253,
                "This run": sum((name.endswith(":m") for name in keep)),
            },
            {
                "Measure": "Recent-minus-prior change features",
                "Source reference": 248,
                "This run": sum((name.endswith(":d") for name in keep)),
            },
            {
                "Measure": "Event rows with no pre-event EPS measurements",
                "Source reference": "Not separately reported",
                "This run": int(
                    (
                        (
                            frame.loc[frame["category"].ne(
                                "Control"), "observed_level_features"]
                        ).eq(
                            0,
                        )
                    ).sum(),
                ),
            },
        ],
    )
    display_table(counts)
    show_message(
        "Genie reproduction: exploratory benchmark",
        (
            "The original global coverage filter, glob"
            "al median imputation, row-stratified fold"
            "s, candidate search and ROC threshold sea"
            "rch are intentionally reproduced here. Th"
            "ey can produce optimistic figures. These "
            "results are not proof of future detection"
            " or a confirmed healthy-engine FPR."
        ),
        "warning",
    )
    columns = [
        "Experiment",
        "Issue",
        "Model",
        "K",
        "Subset",
        "Assessed incidents",
        "Assessed control windows",
        "AUC",
        "Detection at 5% control-window flags",
        "Detection at 10% control-window flags",
        "Detection at 15% control-window flags",
        "Date/history/coverage-only AUC",
    ]
    results = P15["benchmark_results"]
    if not results.empty:
        shown = results.reindex(columns=columns).copy()
        for name in shown:
            if name.startswith("Detection at"):
                shown[name] = shown[name].map(percent_text)
        display_table(shown)
    else:
        show_message(
            "Genie benchmark unavailable",
            (
                "The supplied data lack enough event/c"
                "ontrol observations or usable feature"
                "s for five-fold reproduction. Individ"
                "ual parameter reviews still run."
            ),
            "warning",
        )
    show_message(
        "Controlled comparisons",
        (
            "These use fixed model settings from the s"
            "ource. Engine separation and training-onl"
            "y preprocessing are changed in sequence w"
            "hile retaining the original dates. The ma"
            "tched-date experiment follows. Counts and"
            " missing-evidence coverage are printed so"
            " differing populations are visible."
        ),
    )
    comparison = P15["comparison_results"]
    additional = []
    for head, result in P15["validation"].items():
        if (
            head == "HPT2"
            or result.get("all_predictions", pd.DataFrame()).empty
        ):
            continue
        additional.append(
            roc_profile(
                result["all_predictions"],
                (
                    "Matched dates + engine separation"
                    " + training-only preprocessing"
                ),
                head,
                AUDIT_PRESETS[head],
            ),
        )
    if additional:
        comparison = pd.concat(
            [comparison, pd.DataFrame(additional)],
            ignore_index=True,
        )
    if not comparison.empty:
        shown = comparison.reindex(columns=columns).copy()
        for name in shown:
            if name.startswith("Detection at"):
                shown[name] = shown[name].map(percent_text)
        display_table(shown)
    checks = P15["benchmark_fold_checks"]
    if not checks.empty:
        overlapping = checks["Overlapping train/test engines"].gt(0)
        show_message(
            "Original row-split overlap measured",
            "".join(
                (
                    "{}".format(int(overlapping.sum())),
                    " of ",
                    "{}".format(len(checks)),
                    (
                        " benchmark folds contain an e"
                        "ngine on both training and te"
                        "sting sides. The grouped comp"
                        "arisons and primary incident "
                        "assessments prohibit this ove"
                        "rlap."
                    ),
                ),
            ),
            "warning",
        )
    inventory = P15["feature_inventory"]
    if not inventory.empty:
        display_table(
            (
                inventory.groupby(["Phase", "Selection"]).size()
            ).reset_index(
                name="Columns",
            ),
        )
    download_csv(inventory, "Pearl15_Genie_column_inventory.csv")
    download_csv(
        P15["feature_filter_report"],
        "Pearl15_Genie_feature_coverage.csv",
    )
    download_csv(frame, "Pearl15_Genie_feature_matrix.csv")
    download_csv(
        results,
        "Pearl15_Genie_reproduction_candidates.csv",
    )
    download_csv(
        comparison,
        "Pearl15_Genie_controlled_comparisons.csv",
    )
    download_csv(
        checks,
        "Pearl15_Genie_original_fold_overlap.csv",
    )
    for (head, config), prediction in P15["benchmark_scores"].items():
        download_csv(
            prediction,
            "".join(
                (
                    "Pearl15_Genie_",
                    "{}".format(head),
                    "_",
                    "{}".format(config[0]),
                    "_",
                    "{}".format(config[1]),
                    "_",
                    "{}".format(config[2]),
                    "_predictions.csv",
                ),
            ),
        )


ANALYSIS_OPTIONS = {
    "namespace": NAMESPACE,
    "max_flights": 300,
    "min_flights": 100,
    "lead_days": 0,
    "false_alarm_limit": 0.1,
    "history_start": None,
    "history_end": None,
    "strict_asof": False,
    "takeoff_confirmed": False,
    "eps_causal_verified": False,
    "baseline_confirmed": False,
    "control_cutoff": "2026-10-01",
    "benchmark_grid": True,
    "genie_repeats": 3,
    "tru_repeats": 5,
    "audit_repeats": 3,
    "genie_seed": 42,
    "tru_seed": 7,
    "matched_controls_per_event": 12,
    "eps_match_distance": 0.75,
}
REPORT_OPTIONS = {
    "issues": ("HPT1", "HPT2", "TRU"),
    "engines": (),
    "graph_outputs": ("P50", "T30", "P30"),
    "case_limit": None,
}


def automatic_settings(options):
    settings = dict(DEFAULT_SETTINGS)
    settings.update(
        {
            name: value
            for name, value in options.items()
            if name != "namespace"
        },
    )
    for name in ("genie_repeats", "tru_repeats", "audit_repeats"):
        if not 1 <= int(settings[name]) <= 5:
            raise ValueError(
                "Validation repeats must be between one and five",
            )
    if not 4 <= int(settings["matched_controls_per_event"]) <= 40:
        raise ValueError(
            "Choose four to forty matched controls per event",
        )
    if not 0.1 <= float(settings["eps_match_distance"]) <= 2:
        raise ValueError(
            (
                "History/recency matching tolerance mu"
                "st be between .1 and 2"
            ),
        )
    utc_stamp(settings["control_cutoff"])
    raw = sorted(
        {
            name
            for schema in P15["schemas"].values()
            for name, kind in schema.items()
            if numeric_schema(kind) and raw_measurement(name)
        },
    )
    settings["raw_candidates"] = raw
    settings["inputs"] = default_operating_inputs(raw)
    targets = []
    for family in (
        "P50",
        "T30",
        "P30",
        "VBHP",
        "TGT",
        "FF",
        "OIP",
        "OIT",
        "VBLP",
    ):
        names = [name for name in raw if signal_family(name) == family]
        if names:
            targets.append(names[0])
        elif family in {"P50", "T30", "P30"}:
            targets.append(family)
    settings["targets"] = targets[:8]
    settings["phases"] = [
        phase
        for phase in PHASES
        if (
            (
                "".join(("{}".format(phase), " EPS"))
                in P15["schemas"]
            )
            or (
                "".join(("{}".format(phase), " DA"))
                in P15["schemas"]
            )
        )
    ]
    return validate_settings(settings)


def run_all_analysis(options=None, progress=print):
    options = dict(
        (
            ANALYSIS_OPTIONS
            if options is None
            else {**ANALYSIS_OPTIONS, **options}
        ),
    )
    for name in (
        "validation",
        "raw_reviews",
        "raw_errors",
        "eps_reviews",
        "replays",
        "models",
        "event_models",
        "history_warnings",
    ):
        P15[name] = {}
    P15["cohort"] = None
    for name in (
        "incident_report",
        "case_errors",
        "manifest",
        "eps_groups",
        "genie_matrix",
        "snapshot_data",
    ):
        P15.pop(name, None)
    namespace = options["namespace"].strip()
    tables = {
        name: namespace + "." + path.split(".")[-1]
        for name, path in TABLE_NAMES.items()
    }
    show_message(
        "".join(
            (
                "Automatic Pearl-15 analysis \u00b7 ",
                "{}".format(NOTEBOOK_VERSION),
            ),
        ),
        (
            "The supplied Genie code is reconstructed "
            "first. Controlled checks and independent "
            "engine-held-out incident assessments foll"
            "ow. This is a larger experiment than the "
            "previous notebook; progress is printed as"
            " candidates and repeats finish."
        ),
    )
    display_table(
        discover_catalogues(tables),
        {"Eligible EPS signals": "Not applicable to DA"},
    )
    display_table(P15["schema_report"])
    P15["table_names"] = tables
    settings = automatic_settings(options)
    P15["settings"] = settings
    availability = []
    for phase in settings["phases"]:
        columns = set().union(
            *(
                set(
                    P15["schemas"].get(
                        "".join(
                            (
                                "{}".format(phase),
                                " ",
                                "{}".format(kind),
                            ),
                        ),
                        {},
                    ),
                )
                for kind in ("DA", "EPS")
            ),
        )
        for family in ("P50", "T30", "P30"):
            measured = sorted(
                (
                    name
                    for name in columns
                    if (
                        signal_family(name) == family
                        and raw_measurement(name)
                    )
                ),
            )
            nominal = sorted(
                (
                    name
                    for name in columns
                    if (
                        signal_family(name) == family
                        and re.search("__NOM_|_NOM$", name.upper())
                    )
                ),
            )
            availability.append(
                {
                    "Phase": phase,
                    "Output": family,
                    "Measured columns": ", ".join(measured) or "Unavailable",
                    "Nominal references": ", ".join(nominal) or "None",
                    "Result": (
                        "Measured output available"
                        if measured
                        else (
                            "No actual measurement; no"
                            "minal reference is not pr"
                            "esented as measured outpu"
                            "t"
                        )
                    ),
                },
            )
    display_table(pd.DataFrame(availability))
    display_table(load_catalogues(settings, progress))
    display_table(P15["cycle_report"])
    show_message(
        "Two data counts have different meanings",
        (
            "Classifier features use independently tim"
            "ed EPS records, exactly as the supplied c"
            "ode specifies: a recent mean from up to t"
            "en records, plus recent-minus-prior chang"
            "e. Individual baseline reviews use 100"
            "\u2013300 reconciled take-off anchors. Cr"
            "uise-only EPS can support classification "
            "without inventing flight cycles."
        ),
        "warning",
    )
    build_genie_matrix(settings)
    progress(
        "".join(
            (
                "Genie matrix: ",
                "{}".format(len(P15["genie_matrix"])),
                " assessment rows, ",
                "{}".format(
                    P15["genie_matrix"]["engine"].nunique(),
                ),
                " unique engines, ",
                "{}".format(len(P15["genie_keep"])),
                " retained level/change features",
            ),
        ),
    )
    run_genie_benchmark(settings, progress)
    build_matched_eps_cohort(settings, progress)
    run_reconstruction_comparisons(settings, progress)
    run_reconstructed_validation(settings, progress)
    review_all_events(settings, progress)
    render_genie_reconstruction()
    render_fleet_summary()
    show_message(
        "Automatic reconstruction finished",
        (
            "The reproduction and controlled-compariso"
            "n tables explain the differences. The pri"
            "mary summary uses independently calibrate"
            "d engine-held-out assessments. Following "
            "cells show individual HPT1, HPT2 and TRU "
            "parameter reports. No percentage is copie"
            "d into measured results from the source P"
            "DF."
        ),
    )
    return P15["incident_report"]
