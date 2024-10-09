import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Correlation:
    def __init__(self,
                 x: np.ndarray | pd.Series | pd.DataFrame | List,
                 y: np.ndarray | pd.Series | pd.DataFrame | List,
                 none=False):

        self.x = self._check_input(x)
        self.y = self._check_input(y)

        check = ~np.isnan(self.x) & ~np.isnan(self.y)
        self.x = self.x[check]
        self.y = self.y[check]

        if self.x.shape != self.y.shape:
            raise ValueError(f"x and y must have the same shape (x={self.x.shape}, y={self.y.shape})")

        self.len = len(self.x)
        self.mean_x = self._mean(self.x)
        self.mean_y = self._mean(self.y)
        self.variance_x = self._variance(self.x, self.mean_x)
        self.variance_y = self._variance(self.y, self.mean_y)
        self.std_x = np.sqrt(self.variance_x)
        self.std_y = np.sqrt(self.variance_y)
        self.covariance = self._covariance()
        self.correlation = self._pearson_correlation()
        self.pearson_correlation = self._pearson_correlation()
        self.spearman_correlation = self._spearman_correlation()
        self.contingency_table = self._get_contingency_table(self.x, self.y)
        self.chi2_contingency = self._chi2_contingency()
        self.cramers_v = self._cramers_v()
        self.phi_coefficient = self._phi_coefficient()
        self.point_biserial = self._point_biserial()
        self.kendall_tau = self._kendall_tau()
        self.WoE = self._get_WoE()
        self.view_WoE = self._view_WoE()



    def _check_input(self, data) -> float:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 1:
                print(f"Only the first column is taken from the DataFrame: ({data.shape[1]})")
            data = data.iloc[:, 0]
        return np.array(data)

    def _mean(self, data) -> float:
        return sum(data) / len(data)

    def _variance(self, data, mean) -> float:
        return sum((x - mean) ** 2 for x in data) / (len(data) - 1)

    def _covariance(self) -> float:
        c = sum((xi - self.mean_x) * (yi - self.mean_y) for xi, yi in zip(self.x, self.y)) \
            / (len(self.x) - 1)
        return c

    def _pearson_correlation(self) -> float:
        return self.covariance / (self.std_x * self.std_y)

    def _rank(self, data):
        sort_value = sorted((val, i) for i, val in enumerate(data))
        r = [0] * len(data)
        i = 0
        while i < len(sort_value):
            y, k = sort_value[i]
            same_value = [k]
            j = i + 1
            while j < len(sort_value) and sort_value[j][0] == y:
                same_value.append(sort_value[j][1])
                j += 1
            avg = (i + 1 + j) / 2.0
            for idx in same_value:
                r[idx] = avg
            i = j
        return r

    def _spearman_correlation(self) -> float:
        x = self._rank(self.x)
        y = self._rank(self.y)
        mean_x = self._mean(x)
        mean_y = self._mean(y)
        v_x = self._variance(x, mean_x)
        v_y = self._variance(y, mean_y)
        std_x = np.sqrt(v_x)
        std_y = np.sqrt(v_y)
        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (len(x) - 1) / (std_x * std_y)

    def _get_contingency_table(self, x, y) -> np.array:
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        table = np.zeros((len(unique_x), len(unique_y)), dtype=int)
        for i in range(len(x)):
            row_idx = np.where(unique_x == x[i])[0][0]
            col_idx = np.where(unique_y == y[i])[0][0]
            table[row_idx, col_idx] += 1

        return table
    def _chi2_contingency(self) -> float:
        n = np.sum(self.contingency_table)
        ex = np.zeros_like(self.contingency_table, dtype=float)

        for i in range(len(self.contingency_table)):
            for j in range(len(self.contingency_table[i])):
                ex[i, j] = (np.sum(self.contingency_table[i, :]) * np.sum(self.contingency_table[:, j])) / n
        return np.sum(((self.contingency_table - ex) ** 2) / ex)

    def _cramers_v(self) -> float:
        return np.sqrt(self.chi2_contingency / (self.len * (min(self.contingency_table.shape) - 1)))

    def _phi_coefficient(self) -> float:
        if self.contingency_table.shape != (2, 2):
            return "Phi coefficient is only for binarry data"
        else:
          return (self.contingency_table[0, 0] * self.contingency_table[1, 1] - \
                  self.contingency_table[0, 1] * self.contingency_table[1, 0]) / \
                 np.sqrt((self.contingency_table[0, :] * self.contingency_table[:, 1]).sum() * \
                         (self.contingency_table[0, :] * self.contingency_table[:, 0]).sum())

    def _point_biserial(self):
        y1 = self.y[self.x == 1]
        y0 = self.y[self.x == 0]
        mean_y1 = np.mean(y1)
        mean_y0 = np.mean(y0)
        s_y = np.std(self.y, ddof=1)
        n1 = len(y1)
        n0 = len(y0)
        return ((mean_y1 - mean_y0) / s_y) * np.sqrt((n1 * n0) / self.len**2)

    def _kendall_tau(self):
      c = 0
      d = 0

      for i in range( - 1):
          for j in range(i + 1, self.len):
              if (self.x[i] - self.x[j]) * (self.y[i] - self.y[j]) > 0:
                  c += 1
              elif (self.x[i] - self.x[j]) * (self.y[i] - self.y[j]) < 0:
                  d += 1
      return (c - d) / (0.5 * self.len * (self.len - 1))

    def _get_WoE(self) -> pd.DataFrame:

        df = pd.DataFrame({'x': self.x, 'y': self.y})
        is_binary = df['y'].isin([0, 1]).all()
        if not is_binary:
            return 'Target (y) must be binary'
      
        df = df.groupby('x')['y'].value_counts().unstack(fill_value=0).rename(columns={0: 'No', 1: 'Yes'})
        df['Persentage event'] = df['Yes'] / df['Yes'].sum()
        df['Persentage non event'] = df['No'] / df['No'].sum()
        df['WOE'] = np.log(df['Persentage event'] / df['Persentage non event'].replace(0, np.nan))

        df = df.fillna(0)
        df = df.sort_values('WOE')
        df["rank"] = df['WOE'].rank()
        return df


    def _view_WoE(self) -> go.Figure:

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=self.WoE.index,
            y=self.WoE['WOE'],
            name='WOE'
        ))
        fig.add_trace(go.Scatter(x=self.WoE.index,
                                y=self.WoE['WOE'],
                                name='Woet of Evedence',
                                mode='lines+markers',
                                marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.WoE.index,
                                y=self.WoE['rank'],
                                name='Rank of Evedence',
                                marker=dict(color='green')))
        return fig


    def __str__(self) -> str:
        text = f"""
        Correlation: {self.correlation}
        Pearson correlation: {self.pearson_correlation}
        Spearman correlation: {self.spearman_correlation}
        Phi coefficient: {self.phi_coefficient}
        Cramer's V: {self.cramers_v}
        """
        return text

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, key) -> float:
        if key == "mean_x":
            return self.mean_x
        elif key == "mean_y":
            return self.mean_y
        elif key == "correlation":
            return self.correlation
        elif key == "pearson_correlation":
            return self.pearson_correlation
        elif key == "spearman_correlation":
            return self.spearman_correlation
        elif key == "phi_coefficient":
            return self.phi_coefficient
        elif key == "cramers_v":
            return self.cramers_v
        elif key == "contingency_table":
            return self.contingency_table
        elif key == "chi2_contingency":
            return self.chi2_contingency
        elif key == "point_biserial":
            return self.point_biserial
        elif key == "kendall_tau":
            return self.kendall_tau
        elif key == "WoE":
            return self.WoE
        elif key == "view_WoE":
            return self.view_WoE
        elif key == "x":
            return self.x
        elif key == "y":
            return self.y
        elif key == "variance_x":
            return self.variance_x
        elif key == "variance_y":
            return self.variance_y
        elif key == "std_x":
            return self.std_x
        elif key == "std_y":
            return self.std_y
        elif key == "covariance":
            return self.covariance
        elif key == "len":
            return self.len
        else:
            raise ValueError(f"Invalid key: {key}")
