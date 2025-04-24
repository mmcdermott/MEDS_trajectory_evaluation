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
>>> from aces.config import PlainPredicateConfig, EventConfig, TaskExtractorConfig, WindowConfig

```

We'll also import the `print_ACES` helper function to visualize the task configs:

```python
>>> from zeroshot_ACES.aces_utils import print_ACES

```

#### Example 1: In-hospital mortality prediction

**TODO**

#### Example 2: 30-day post discharge mortality prediction

Given a hospital admission, we'll use the first 24 hours of data to predict whether or not the patient will
die within 30 days of discharge (with a 1-day gap window post discharge to avoid future leakage):

```python
>>> predicates = {
...     "admission": PlainPredicateConfig("ADMISSION"),
...     "discharge": PlainPredicateConfig("DISCHARGE"),
... }
>>> trigger = EventConfig("admission")
>>> windows = {
...     "input": WindowConfig(None, "trigger + 24h", True, True),
...     "discharge": WindowConfig("input.end", "start -> discharge", False, True),
...     "target": WindowConfig("discharge.end + 1d", "start + 29d", False, True),
... }
>>> post_discharge_cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
>>> print_ACES(post_discharge_cfg)
    trigger
    └── (+1 day, 0:00:00) input.end
        ├── (prior _RECORD_START) input.start
        └── (next discharge) discharge.end
            └── (+1 day, 0:00:00) target.start
                └── (+29 days, 0:00:00) target.end

```

#### Example 3: 30-day readmission prediction with censoring

**TODO**\*

#### Example 4: ???

### Relaxations

#### 1. Remove inclusion/exclusion criteria

This relaxation removes all inclusion/exclusion criteria from the task config, but does not change the window
boundaries that are used to compile the task cohort.

> [!NOTE]
> Using this relaxation does _not_ mean that predictions are made over task samples that failed to meet the
> task criteria (with respect to their real data). Rather, it just means that generated trajectories will not
> be discarded on the basis of failing to meet post-input window inclusion/exclusion criteria.

This relaxation can be applied in several variants:

1. Remove all inclusion/exclusion criteria.
2. Remove all post-target window inclusion/exclusion criteria (e.g., remove censoring protections).

#### 2. Absorb gap windows into target

This relaxation absorbs all windows between the input and target windows into the target window. This can only
be used if the constraints of these windows are all identical and mutually satisfiable. This relaxation allows
you to make predictions with fewer generated tokens and simpler early stopping criteria.

> [!NOTE]
> This relaxation does not change the total time-span expected for the target window; it merely extends the
> target window's start earlier in time to be the end of the input window, where possible.
