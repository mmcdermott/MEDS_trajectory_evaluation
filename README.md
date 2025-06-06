# MEDS Trajectory Evaluation

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS_trajectory_evaluation)](https://pypi.org/project/MEDS_trajectory_evaluation/)
![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![codecov](https://codecov.io/gh/mmcdermott/MEDS_trajectory_evaluation/graph/badge.svg?token=CPLS7DPPAK)](https://codecov.io/gh/mmcdermott/MEDS_trajectory_evaluation)
[![tests](https://github.com/mmcdermott/MEDS_trajectory_evaluation/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_trajectory_evaluation/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/MEDS_trajectory_evaluation/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_trajectory_evaluation/actions/workflows/code-quality-main.yaml)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MEDS_trajectory_evaluation#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MEDS_trajectory_evaluation/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/MEDS_trajectory_evaluation.svg)](https://github.com/mmcdermott/MEDS_trajectory_evaluation/graphs/contributors)

This package contains utilities for converting autoregressive, generated trajectories into probabilistic
predictions for arbitrary ACES configuration files.

## 1. Install

```bash
pip install MEDS_trajectory_evaluation
```

## 2. Run

```bash
ZSACES_label task.criteria_fp="$TASK_CRITERIA" task.predicates_fp="$PREDICATES_FP" \
    output_dir=$OUTPUT_DIR trajectories_dir=$TRAJECTORIES_DIR
```

Optionally, you can add relaxations to the zero-shot labeling config via `labeler.remove_all_criteria=True`,
`labeler.collapse_temporal_gap_windows=True`, or `labeler.remove_post_label_windows=true`. See below for
examples of these in action.

# Documentation

> [!IMPORTANT]
> This library only works with a subset of ACES configs; namely, those that have a tree-based set of
> dependencies between the end of the input window (the prediction time) and the end of the target window (the
> label window).

## Terminology

| Term                            | Description                                                                                                                                                                                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACES                            | [ACES](https://eventstreamaces.readthedocs.io/en/latest/) is a domain specific language for describing task cohorts and a tool to automatically extract them from EHR datasets. It is the "source of truth" for task definitions in this work.                                         |
| Task Config                     | The (original/raw) ACES configuration file that describes the task cohort.                                                                                                                                                                                                             |
| Input Window                    | The window in the ACES config defining the "prediction time". This is indicated via the `index_timestamp` marker in the ACES config.                                                                                                                                                   |
| Target Window                   | The window in the ACES config over which the label is extracted. This is indicated via the `label` marker in the ACES config.                                                                                                                                                          |
| Normal-form / Normalized Config | When in "normal-form" or "normalized", a config has an input window that ends with the prediction time and the prediction time node in the task config tree is an ancestor of both ends of the target window.                                                                          |
| Relaxations                     | A configuration relaxation is a modification to the task config that removes constraints or simplifies the relationships between window endpoints. These are used to simplify or broaden the set of identified empricial labels during zero-shot prediction vs. task label extraction. |
| valid                           | A trajectory is "valid" under a config when it does not indicate a sequence of measurements that would violate any inclsuion/exclusion criteria in the zero-shot task config.                                                                                                          |
| determinable                    | A trajectory is "determinable" under a config if and only if it is both valid and contains valid realizations of all relevant windows in the config (e.g., we don't need to generate more).                                                                                            |

## Supported Config Relaxations

We support a few different relaxations that can help make zero-shot label extraction simpler and more
accommodating. These relaxations are not always appropriate for all tasks, but they can be useful in some
cases. To understand them deeply, we'll use several examples, which we'll set up first.

### Example Configurations

To explore these relaxations, we'll use a few simple example task configs. To construct them, we first need to
import the relevant ACES config classes:

```python
>>> from aces.config import (
...     PlainPredicateConfig, EventConfig, TaskExtractorConfig, WindowConfig, DerivedPredicateConfig,
... )

```

We'll also import the `print_ACES` helper function to visualize the task configs:

```python
>>> from MEDS_trajectory_evaluation.aces_utils import print_ACES

```

#### Example 1: In-hospital mortality prediction

```python
>>> in_hosp_mortality_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
...         "input": WindowConfig(
...             "trigger", "start + 24h", False, True, index_timestamp="end",
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "gap": WindowConfig(
...             "input.end", "start + 24h", False, True,
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True, label="death"),
...     }
... )
>>> print_ACES(in_hosp_mortality_cfg)
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
        └── (next discharge_or_death) target.end; **Label: Presence of death**

```

#### Example 2: 30-day post discharge mortality prediction

Given a hospital admission, we'll use the first 24 hours of data to predict whether or not the patient will
die within 30 days of discharge (with a 1-day gap window post discharge to avoid future leakage). We'll also
impose another gap window after the admission to ensure that the hospitalization itself lasts at least 48
hours.

```python
>>> post_discharge_mortality_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
...         "input": WindowConfig(
...             "trigger", "start + 24h", False, True, index_timestamp="end",
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "post_input": WindowConfig(
...             "input.end", "start + 1d", False, True,
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "hospitalization": WindowConfig(
...             "input.end", "start -> discharge", False, True, has={"death": "(None, 0)"},
...         ),
...         "gap": WindowConfig(
...             "hospitalization.end", "start + 1d", False, True,
...             has={"admission": "(None, 0)", "death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start + 29d", False, True, label="death"),
...     }
... )
>>> print_ACES(post_discharge_mortality_cfg)
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    ├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
    └── (next discharge) hospitalization.end (no death)
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

#### Example 3: 30-day readmission prediction with censoring

This example features a 30-day readmission risk prediction task, but with a post-target censoring protection
window.

```python
>>> readmission_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("discharge"),
...     windows={
...         "sufficient_history": WindowConfig(
...             None, "hospitalization.start", True, False, has={"_ANY_EVENT": "(5, None)"}
...         ),
...         "hospitalization": WindowConfig(
...             "end <- admission", "trigger", True, True, has={"_ANY_EVENT": "(10, None)"},
...             index_timestamp="end"
...         ),
...         "gap": WindowConfig(
...             "hospitalization.end", "start + 1d", False, True,
...             has={"admission": "(None, 0)", "death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start + 29d", False, True, label="admission"),
...         "censoring_protection": WindowConfig(
...             "target.end", None, True, True, has={"_ANY_EVENT": "(1, None)"},
...         ),
...     }
... )
>>> print_ACES(readmission_cfg)
trigger; **Prediction Time**
├── (prior admission) hospitalization.start (at least 10 event(s))
│   └── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))

```

#### Example 4: Two-stage Infusion

In this hypothetical example, we are examining a cohort of patients who are given an infusion, then given a
drug, then (within 10 minutes) have their infusion stopped temporarily, then resumed. We are interested in
predicting, at the time of the drug being given, about an adverse event within their second infusion stage.
The reason to have such a task is to explore when relaxations are or aren't appropriate in more complex
set-ups.

```python
>>> two_stage_cfg = TaskExtractorConfig(
...     predicates={
...         "infusion_start": PlainPredicateConfig("INFUSION//START"),
...         "infusion_end": PlainPredicateConfig("INFUSION//END"),
...         "drug_given": PlainPredicateConfig("special_drug"),
...         "adverse_event": PlainPredicateConfig("special_adverse_event"),
...     },
...     trigger=EventConfig("drug_given"),
...     windows={
...         "1st_infusion": WindowConfig(
...             "trigger", "start -> infusion_end", True, True, has={"adverse_event": "(None, 0)"},
...             index_timestamp="start",
...         ),
...         "2nd_infusion": WindowConfig(
...             "1st_infusion.end -> infusion_start", "start -> infusion_end", True, True,
...             label="adverse_event"
...         ),
...     }
... )
>>> print_ACES(two_stage_cfg)
trigger; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

#### Other examples we can't reflect:

1. What if we only want to count something as a readmission only if the next admission has a discharge
    associated with a particular diagnosis code? We can't reflect this in ACES currently, but it would pose
    additional challenges.

### Relaxations

We can perform any of the relaxations with the `convert_to_zero_shot` function in
[`task_config`](src/MEDS_trajectory_evaluation/ACES_config_evaluation/task_config.py) and an appropriate labeler config. Let's import that now for
use with our examples:

```python
>>> from MEDS_trajectory_evaluation.ACES_config_evaluation.task_config import convert_to_zero_shot

```

Even without any relaxations, the zero-shot conversion will naturally prunes the tree to include only those
nodes between the prediction time window and the label window or after the label window.

```python
>>> print_ACES(convert_to_zero_shot(in_hosp_mortality_cfg))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

> [!WARNING]
> This can remove some criteria that you may still want to leverage. See, for example, how the post discharge
> config has lost the window asserting the hospitalization is at least 48 hours. This could be corrected by
> having the hospitalization window depend directly on the post input window, rather than the input window.

```python
>>> print_ACES(convert_to_zero_shot(post_discharge_mortality_cfg))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end (no death)
    └── (+1 day, 0:00:00) gap.end (no admission, death)
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

We still retain the prediction time, label, and relevant criteria in this view.

```python
>>> print_ACES(convert_to_zero_shot(readmission_cfg))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))
>>> print_ACES(convert_to_zero_shot(two_stage_cfg))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

#### 1. `remove_all_criteria`: Remove inclusion/exclusion criteria

This relaxation removes all inclusion/exclusion criteria from the task config, but does not change the window
boundaries that are used to compile the task cohort.

> [!NOTE]
> Using this relaxation does _not_ mean that predictions are made over task samples that failed to meet the
> task criteria (with respect to their real data). Rather, it just means that generated trajectories will not
> be discarded on the basis of failing to meet post-input window inclusion/exclusion criteria.

##### On Example 1: In Hospital Mortality

```python
>>> print_ACES(convert_to_zero_shot(in_hosp_mortality_cfg, {"remove_all_criteria": True}))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

Here, this may be a mistake, as it will classify trajectories as true if they die after discharge, provided
discharge is within 1 day. However, using this in conjunction with absorbing gap windows is likely suitable.

##### On Example 2: Post-discharge Mortality

```python
>>> print_ACES(convert_to_zero_shot(post_discharge_mortality_cfg, {"remove_all_criteria": True}))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end
    └── (+1 day, 0:00:00) gap.end
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

Here, this is may be a mistake, as it will classify as negative trajectories who die within 1 day after
discharge (whereas previously such trajectories would be excluded). However, in concert with gap window
absorption, this may be suitable.

##### On Example 3: Readmission

```python
>>> print_ACES(convert_to_zero_shot(readmission_cfg, {"remove_all_criteria": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end

```

In this example, there are both good and bad aspects of these changes. First, this will now label trajectories
as negative if they are admitted within 1 day (previously, they would have been excluded), which is likely
problematic. But it also renders the censoring window moot, which may improve the efficiency.

##### On Example 4: 2nd infusion stage adverse event

```python
>>> print_ACES(convert_to_zero_shot(two_stage_cfg, {"remove_all_criteria": True}))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This may be suitable here; it still tracks the right target (adverse events within the 2nd infusion period),
but now will include labels for patients who have adverse events in both, which may improve the predictive
quality or efficiency of the trajectory-driven predictor.

#### 2. `collapse_temporal_gap_windows`: Absorb temporal gap windows into target

This relaxation absorbs any chain of temporal windows between the input and target window terminating at the
target window into the target window. This can only be used if the constraints of these windows are all
removed (or if the remove all criteria relaxation is applied as well). This relaxation allows you to make
predictions with fewer generated tokens and simpler early stopping criteria.

> [!NOTE]
> This does not remove event bounded windows, though it does remove temporal windows directly before event
> bound windows or absorb adjacent temporal windows together.

```python
>>> labeler_cfg = {"remove_all_criteria": True, "collapse_temporal_gap_windows": True}

```

##### On Example 1: In Hospital Mortality

```python
>>> print_ACES(convert_to_zero_shot(in_hosp_mortality_cfg, labeler_cfg))
input.end; **Prediction Time**
└── (next discharge_or_death) target.end; **Label: Presence of death**

```

This is likely appropriate, as we will now simply classify if there is any death observed before the next
discharge.

##### On Example 2: Post-discharge Mortality

```python
>>> print_ACES(convert_to_zero_shot(post_discharge_mortality_cfg, labeler_cfg))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end
    └── (+30 days, 0:00:00) target.end; **Label: Presence of death**

```

This is likely suitable; we have simply stremlined the prediction target to be anytime within the 30 days post
discharge, giving the trajectory labeler a more flexible target.

##### On Example 3: Readmission

```python
>>> print_ACES(convert_to_zero_shot(readmission_cfg, labeler_cfg))
hospitalization.end; **Prediction Time**
└── (+30 days, 0:00:00) target.end; **Label: Presence of admission**
    └── (end of record) censoring_protection.end

```

This is likely an improvement over the basic config, because it is more accommodating to the target, but it
still has a censoring prediction window we may want to remove.

##### On Example 4: 2nd infusion stage adverse event

```python
>>> print_ACES(convert_to_zero_shot(two_stage_cfg, labeler_cfg))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This makes no difference as there are no temporal gap windows in this example.

#### 3. `remove_post_label_windows`: Removes all post-label windows from the task config

This relaxation removes all windows that are after the label window. This is useful for removing censoring
protection windows which expand the generation scope necessary to resolve a window.

##### On Example 1: In Hospital Mortality

```python
>>> print_ACES(convert_to_zero_shot(in_hosp_mortality_cfg, {"remove_post_label_windows": True}))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

This makes no difference as there are no post-label windows in this example.

##### On Example 2: Post-discharge Mortality

```python
>>> print_ACES(convert_to_zero_shot(post_discharge_mortality_cfg, {"remove_post_label_windows": True}))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end (no death)
    └── (+1 day, 0:00:00) gap.end (no admission, death)
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

This makes no difference as there are no post-label windows in this example.

##### On Example 3: Readmission

```python
>>> print_ACES(convert_to_zero_shot(readmission_cfg, {"remove_post_label_windows": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**

```

This is likely an improvement, as the censoring protection may complicate generation and reduce the
efficiency.

##### On Example 4: 2nd infusion stage adverse event

```python
>>> print_ACES(convert_to_zero_shot(two_stage_cfg, {"remove_post_label_windows": True}))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This makes no difference as there are no post-label windows in this example.

### Examples of Labeling

To see labeling in action, we'll work with the following configuration:

```python
>>> print_ACES(sample_ACES_cfg)
trigger
└── (+1 day, 0:00:00) input.end (no icu_admission, discharge_or_death); **Prediction Time**
    └── (+1 day, 0:00:00) gap.end (no icu_admission, discharge_or_death)
        └── (next discharge_or_death) target.end; **Label: Presence of death**

```

We'll also use the following generated trajectories:

```python
>>> for fn, df in sample_labeled_trajectories_dfs.items():
...     print(f"Generated trajectory: {fn}")
...     print(df)
Generated trajectory: trajectory_0.parquet
shape: (9, 5)
┌─────────────────────────┬───────────────┬───────────────┬────────────┬─────────────────────────┐
│ time                    ┆ code          ┆ numeric_value ┆ subject_id ┆ prediction_time         │
│ ---                     ┆ ---           ┆ ---           ┆ ---        ┆ ---                     │
│ datetime[μs, UTC]       ┆ str           ┆ f64           ┆ i32        ┆ datetime[μs, UTC]       │
╞═════════════════════════╪═══════════════╪═══════════════╪════════════╪═════════════════════════╡
│ 1993-01-01 12:00:00 UTC ┆ LAB_1         ┆ 1.0           ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-01-01 13:00:00 UTC ┆ LAB_2         ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-01-01 14:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-01-22 00:00:00 UTC ┆ MEDS_DEATH    ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-02-20 00:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
│ 1995-01-01 00:00:00 UTC ┆ LAB_23        ┆ 1.2           ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
│ 1999-01-01 13:00:00 UTC ┆ LAB_3         ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
│ 1999-01-01 14:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
│ 1999-01-04 14:00:00 UTC ┆ LAB_4         ┆ 1.1           ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
└─────────────────────────┴───────────────┴───────────────┴────────────┴─────────────────────────┘
Generated trajectory: trajectory_1.parquet
shape: (6, 5)
┌─────────────────────────┬───────────────┬───────────────┬────────────┬─────────────────────────┐
│ time                    ┆ code          ┆ numeric_value ┆ subject_id ┆ prediction_time         │
│ ---                     ┆ ---           ┆ ---           ┆ ---        ┆ ---                     │
│ datetime[μs, UTC]       ┆ str           ┆ f64           ┆ i32        ┆ datetime[μs, UTC]       │
╞═════════════════════════╪═══════════════╪═══════════════╪════════════╪═════════════════════════╡
│ 1993-01-01 12:00:00 UTC ┆ LAB_1         ┆ 1.0           ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-01-04 00:00:00 UTC ┆ MEDS_DEATH    ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1998-01-01 00:00:00 UTC ┆ LAB_1         ┆ 1.1           ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
│ 2000-01-01 00:00:00 UTC ┆ LAB_3         ┆ 1.2           ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
│ 1999-01-01 12:00:00 UTC ┆ ICU_ADMISSION ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
│ 1999-02-01 00:00:00 UTC ┆ MEDS_DEATH    ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
└─────────────────────────┴───────────────┴───────────────┴────────────┴─────────────────────────┘
Generated trajectory: trajectory_2.parquet
shape: (3, 5)
┌─────────────────────────┬───────────────┬───────────────┬────────────┬─────────────────────────┐
│ time                    ┆ code          ┆ numeric_value ┆ subject_id ┆ prediction_time         │
│ ---                     ┆ ---           ┆ ---           ┆ ---        ┆ ---                     │
│ datetime[μs, UTC]       ┆ str           ┆ null          ┆ i32        ┆ datetime[μs, UTC]       │
╞═════════════════════════╪═══════════════╪═══════════════╪════════════╪═════════════════════════╡
│ 1993-01-01 12:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 1993-01-01 13:00:00 UTC ┆ ICU_ADMISSION ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
│ 2005-01-01 00:00:00 UTC ┆ MEDS_DEATH    ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
└─────────────────────────┴───────────────┴───────────────┴────────────┴─────────────────────────┘

```

What labels do we get if we run the labeling function on these with various relaxations of our config? To see,
first we need to import the label function:

```python
>>> from MEDS_trajectory_evaluation.ACES_config_evaluation.label import label_trajectories

```

#### 1. No Relaxations

```python
>>> print_ACES(convert_to_zero_shot(sample_ACES_cfg))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no icu_admission, discharge_or_death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**
>>> for fn, df in sample_labeled_trajectories_dfs.items():
...     print(f"Labels for {fn}:")
...     print(label_trajectories(df, convert_to_zero_shot(sample_ACES_cfg)))
Labels for trajectory_0.parquet:
shape: (3, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ false ┆ null         ┆ null  │
│ 1          ┆ 1993-01-20 00:00:00 UTC ┆ true  ┆ true         ┆ false │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ false ┆ null         ┆ null  │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘
Labels for trajectory_1.parquet:
shape: (3, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ true  │
│ 1          ┆ 1993-01-20 00:00:00 UTC ┆ true  ┆ false        ┆ null  │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ false ┆ null         ┆ null  │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘
Labels for trajectory_2.parquet:
shape: (2, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ false ┆ null         ┆ null  │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ true  │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘

```

#### 2. Without gap windows or criteria

```python
>>> labeler_cfg = {"remove_all_criteria": True, "collapse_temporal_gap_windows": True}
>>> print(f"Under labeler_cfg={labeler_cfg}")
Under labeler_cfg={'remove_all_criteria': True, 'collapse_temporal_gap_windows': True}
>>> print_ACES(convert_to_zero_shot(sample_ACES_cfg, labeler_cfg))
input.end; **Prediction Time**
└── (next discharge_or_death) target.end; **Label: Presence of death**
>>> for fn, df in sample_labeled_trajectories_dfs.items():
...     print(f"Labels for {fn}:")
...     print(label_trajectories(df, convert_to_zero_shot(sample_ACES_cfg, labeler_cfg)))
Labels for trajectory_0.parquet:
shape: (3, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ false │
│ 1          ┆ 1993-01-20 00:00:00 UTC ┆ true  ┆ true         ┆ false │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ false │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘
Labels for trajectory_1.parquet:
shape: (3, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ true  │
│ 1          ┆ 1993-01-20 00:00:00 UTC ┆ true  ┆ false        ┆ null  │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ true  │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘
Labels for trajectory_2.parquet:
shape: (2, 5)
┌────────────┬─────────────────────────┬───────┬──────────────┬───────┐
│ subject_id ┆ prediction_time         ┆ valid ┆ determinable ┆ label │
│ ---        ┆ ---                     ┆ ---   ┆ ---          ┆ ---   │
│ i32        ┆ datetime[μs, UTC]       ┆ bool  ┆ bool         ┆ bool  │
╞════════════╪═════════════════════════╪═══════╪══════════════╪═══════╡
│ 1          ┆ 1993-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ false │
│ 2          ┆ 1999-01-01 00:00:00 UTC ┆ true  ┆ true         ┆ true  │
└────────────┴─────────────────────────┴───────┴──────────────┴───────┘

```

## Temporal AUC Evaluation

The `temporal_AUC_evaluation` package contains helpers for turning
time-to-first-event observations into AUC summaries across multiple prediction
horizons.

### Helper functions

- `get_raw_tte` and `get_trajectory_tte` extract time-to-event values for each
    predicate from real datasets or generated trajectories.
- `merge_pred_ttes` stacks multiple predicted TTE tables into list columns so
    probability distributions can be derived per subject.
- `add_labels_from_true_tte` converts true durations into binary labels for a
    given horizon and `add_probs_from_pred_ttes` turns predicted durations into
    probabilities of observing the event within that window.

### Computing AUCs

`temporal_aucs` wires these pieces together and returns a DataFrame indexed by
duration with `AUC/<predicate>` columns detailing discrimination for each
predicate at every horizon.

```python
>>> temporal_aucs(true_tte_df, pred_tte_df, [timedelta(days=1), timedelta(days=7)])  # doctest: +SKIP
shape: (2, 3)
┌──────────────┬────────┬────────┐
│ duration     ┆ AUC/A  ┆ AUC/B │
│ ---          ┆ ---    ┆ ---   │
│ duration[μs] ┆ f64    ┆ f64   │
╞══════════════╪════════╪═══════╡
│ 1d           ┆ 0.65   ┆ 0.72  │
│ 7d           ┆ 0.71   ┆ 0.80  │
└──────────────┴────────┴────────┘
```
