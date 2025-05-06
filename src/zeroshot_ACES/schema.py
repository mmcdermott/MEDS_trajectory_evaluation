import pyarrow as pa
from flexible_schema import Required
from meds import DataSchema


class GeneratedTrajectorySchema(DataSchema):
    """Schema for generated MEDS trajectories.

    This extends the MEDS schema by including the `prediction_time` field used to localize the generated
    trajectory to a particular task sample.
    """

    prediction_time: Required(pa.timestamp("us"), nullable=False)
