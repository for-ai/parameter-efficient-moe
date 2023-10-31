import re
from typing import Sequence, Callable, Any

def match_any(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
  """A traversal that checks if the parameter name matches any regex.

  This is returns a closure over the actual traversal function that takes the
  parameter name and value. The return value of this should be in input to the
  Traversal used in the MultiOptimizer.

  Args:
    regexes: A list of regular expressions that denote which parameter should be
      updated by this optimizer.

  Returns:
    A function that takes the name and value of a parameter and return True if
    that parameter should be updated by the optimizer.
  """
  regexes = tuple(re.compile(regex) for regex in regexes)

  def _match_any(path, _):
    """True if path matches any regex in regexs, false otherwise."""
    return any(regex.fullmatch(path) for regex in regexes)

  return _match_any