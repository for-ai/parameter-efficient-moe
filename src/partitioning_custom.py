"""Custom partitioning rules."""

from t5x import partitioning


def standard_logical_axis_rules() -> partitioning.LogicalAxisRules:
  """Add specific partitioning rules."""
  return (
          ("unmodeled", None), 
          ("rank", None),
          ("expert", None),
          )