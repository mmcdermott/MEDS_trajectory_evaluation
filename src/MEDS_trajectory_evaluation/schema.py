"""Schema definition for generated MEDS trajectories.

Extends the core MEDS data schema with a ``prediction_time`` column to associate generated events with the
input data cutoff time used for generation.
"""

import pyarrow as pa
from flexible_schema import Required
from meds import DataSchema


class GeneratedTrajectorySchema(DataSchema):
    """Schema for generated MEDS trajectories.

    This extends the MEDS data schema by including a ``prediction_time`` field that indicates the
    latest time at which input data was used to generate the trajectory. Different trajectory samples
    from the same (subject, prediction_time) pair should be stored as separate DataFrames to avoid
    sort-order ambiguity.

    Attributes:
        prediction_time: Microsecond-resolution timestamp marking the input data cutoff.
            Must not be null.
    """

    prediction_time: Required(pa.timestamp("us"), nullable=False)
