from typing import Optional, Dict, Tuple

import dynet as dy
import numpy as np

class LossExpr(object):
  def __init__(self, expr, units):
    self.expr = expr
    self.units = np.array(units)
    
  def loss_value(self):
    return self.expr, np.sum(self.units)
    
  def __add__(self, other):
    if type(other) == LossExpr:
      new_expr = self.expr + other.expr
      new_units = self.units + other.units
    else:
      new_expr = self.expr + other
      new_units = self.units
    return LossExpr(new_expr, new_units)
    
  def value(self):
    return self.expr.value()

class FactoredLossExpr(object):
  
  """
  Loss consisting of (possibly batched) DyNet expressions, with one expression per loss factor.

  Used to represent losses within a training step.

  Args:
    init_loss: initial loss values
  """

  def __init__(self, init_loss: Optional[Dict[str, LossExpr]] = None) -> None:
    self.expr_factors = {}
    if init_loss is not None:
      for key, value in init_loss.items():
        assert type(value) == LossExpr or type(value) == FactoredLossExpr
        self._add_data(key, value)
    
  def _add_data(self, now_key, value):
    if type(value) is FactoredLossExpr:
      for k, v in value.expr_factors.items():
        self._add_data("{}:{}".format(now_key, k), v)
    elif now_key in self.expr_factors:
      self.expr_factors[now_key] += value
    else:
      self.expr_factors[now_key] = value
    
  def compute(self, comb_method: str = "sum") -> Tuple[dy.Expression, Dict]:
    """
    Compute loss as DyNet expression by summing over factors and batch elements.

    Args:
      comb_method: method for combining loss across batch elements ('sum' or 'avg').

    Returns:
      Scalar DyNet expression.
    """
    loss_exprs = 0
    loss_data = {}
    
    for name, loss_expr in self.expr_factors.items():
      expr, units = loss_expr.loss_value()
      loss_exprs += expr
      loss_data[name] = dy.sum_batches(expr).value(), units
   
    # Combining
    if comb_method == "sum":
      loss_exprs = dy.sum_batches(loss_exprs)
    elif comb_method == "avg":
      loss_exprs = dy.sum_batches(loss_exprs) * (1.0 / loss_exprs.dim()[1])
    else:
      raise ValueError(f"Unknown batch combination method '{comb_method}', expected 'sum' or 'avg'.'")
    
    return loss_exprs, loss_data

  def __getitem__(self, loss_name: str) -> LossExpr:
    return self.expr_factors[loss_name]

  def __mul__(self, scalar):
    return FactoredLossExpr({key: LossExpr(lexpr.expr * scalar, lexpr.units) for key, lexpr \
                             in self.expr_factors.items()})
  
  def __add__(self, other):
    assert type(other) == FactoredLossExpr
    new_expr = FactoredLossExpr()
    new_expr.expr_factors = self.expr_factors
    
    for key, value in other.items():
      new_expr._add_data(key, value)
    return new_expr
  
  def items(self):
    return self.expr_factors.items()
