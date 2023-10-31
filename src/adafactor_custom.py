"""Custom adafactor rules."""

from flax.core.frozen_dict import freeze
from flax.core.frozen_dict import unfreeze
from t5x import adafactor


def standard_logical_factor_rules(rules=None):
  """Add prompt adafactor rules to your set of rules."""
  if rules is None:
    rules = adafactor.standard_logical_factor_rules()
  rules = unfreeze(rules)
  rules['unmodeled'] = adafactor.FactorDim.NONE
  rules['rank'] = adafactor.FactorDim.NONE
  rules['expert'] = adafactor.FactorDim.NONE
  return freeze(rules)