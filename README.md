# Zero-shot ACES

This package contains utilities for converting autoregressive, generated trajectories into probabilistic
predictions for arbitrary ACES configuration files.

## 1. Install

```bash
pip install zeroshot_ACES
```

## 2. Run

```bash
ZSACES_predict ...
```

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
>>> from zeroshot_ACES.aces_utils import print_ACES

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
[`task_config`](src/zeroshot_ACES/task_config.py) and an appropriate labeler config. Let's import that now for
use with our examples:

```python
>>> from zeroshot_ACES.task_config import convert_to_zero_shot

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
