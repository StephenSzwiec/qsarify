from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ModelResult:
    """
    A dataclass to store the results of a single QSAR model.
    """
    model: Any
    name: str
    selected_features: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    y_true_train: Optional[pd.Series] = None
    y_pred_train: Optional[pd.Series] = None
    y_true_test: Optional[pd.Series] = None
    y_pred_test: Optional[pd.Series] = None

    def __str__(self):
        return f"ModelResult(name='{self.name}', R2_train={self.statistics.get('R2_train', 'N/A')}, R2_test={self.statistics.get('R2_test', 'N/A')})"

    def __repr__(self):
        return self.__str__()
