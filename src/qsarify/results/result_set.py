from typing import List

import pandas as pd

from .model_result import ModelResult


class ResultSet:
    """
    A collection of ModelResult objects.
    """

    def __init__(self, results: List[ModelResult] = None):
        self.results = results if results is not None else []

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def append(self, result: ModelResult):
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame summarizing the results.
        """
        records = []
        for res in self.results:
            record = {
                'name': res.name,
                'n_features': len(res.selected_features) if res.selected_features else 0,
                **res.statistics
            }
            records.append(record)
        return pd.DataFrame(records)

    def sort_by(self, metric: str, ascending: bool = False):
        """
        Sorts the results by a given metric.
        """
        self.results.sort(
            key=lambda r: r.statistics.get(metric, -1),
            reverse=not ascending
        )

    def get_best_model(self, metric: str) -> ModelResult:
        """
        Returns the best model based on a given metric.
        """
        if not self.results:
            raise ValueError("ResultSet is empty.")

        return max(self.results, key=lambda r: r.statistics.get(metric, -1))
