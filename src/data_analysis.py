from typing import Literal
import pandas
import seaborn as sns
import matplotlib.pylab as plt
import statsmodels.api as sm

from src.utils import separate_feature_target, write_to_file


class DatasetAnalysis:
    "class that analyzes a dataset"

    def __init__(
        self, data_frame: pandas.core.frame.DataFrame, target: list[str]
    ) -> None:

        self.data_frame = data_frame
        self.target = target

    def export_dataset_describe(
        self,
        decimal_digits: int = 4,
        export_format: Literal[".txt", ".xlsx"] = ".txt",
        path: str = "texts/dataset_description",
    ) -> None:

        if export_format == ".xlsx":
            self.data_frame.describe().round(decimal_digits).to_excel(
                path + export_format, "dataset-description"
            )
        elif export_format == ".txt":
            # Display the table without replacing middle rows and columns with "...".
            with pandas.option_context(
                "display.max_columns", None, "display.max_rows", None
            ):
                write_to_file(
                    self.data_frame.describe().round(decimal_digits),
                    path=path + export_format,
                )

    def export_corr_heatmap(
        self,
        dpi: int = 200,
        saving_path: str = "figs/correlation.png",
        title: str = "The heatmap of variable correlations",
    ) -> None:

        correlations = self.data_frame.corr()
        sns.heatmap(correlations).set(title=title)
        plt.savefig(saving_path, dpi=dpi)

    def export_scatter(
        self,
        dpi: int = 200,
        saving_path: str = "figs/full_scatterplot.png",
        title: str = "The scatter plot of dataset",
    ) -> None:

        g = sns.PairGrid(self.data_frame)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        g.fig.subplots_adjust(top=0.93)
        g.fig.suptitle(title, size=50)
        plt.savefig(saving_path, dpi=dpi)

    def export_most_correlated_scatter(
        self,
        var_num: int = 5,
        font_size: int = 20,
        dpi: int = 200,
        saving_path: str = "figs/correlated_scatterplot.png",
        title: str = "Scatter plot of targets ({}) and {} most correlated variables",
    ) -> None:

        correlations = self.data_frame.corr()
        target_count = len(self.target)
        fig, axes = plt.subplots(
            len(self.target), var_num, figsize=((var_num - 2) * 5, 5)
        )
        axes = [axes] if target_count == 1 else axes
        for i, t in enumerate(self.target):
            most_correlated_vars = (
                correlations[t].drop(t).abs().sort_values().iloc[-var_num:].index
            )
            for ax, var in zip(axes[i], most_correlated_vars):
                sns.scatterplot(x=self.data_frame[var], y=self.data_frame[t], ax=ax)
                ax.set_title(f"corr({var}, {t})={correlations[var][t]:.2f}")
        fig.suptitle(title.format(", ".join(self.target), var_num), fontsize=font_size)
        fig.tight_layout()
        plt.savefig(saving_path, dpi=dpi)

    def export_p_value_calculation(
        self, path: str = "texts/p_value_calculations.txt"
    ) -> None:

        features, targets = separate_feature_target(self.data_frame, self.target)
        augmented_dataset = sm.add_constant(features)
        xname = ["constant"] + [
            c for c in self.data_frame.columns if c not in self.target
        ]
        for i, t in enumerate(self.target):
            linear_model = sm.OLS(targets[:, i], augmented_dataset).fit()
            file_mode = "w" if i == 0 else "a"
            write_to_file(
                f"p-values for target {t}:",
                linear_model.summary(yname=t, xname=xname),
                "\n" * 3,
                path=path,
                mode=file_mode,
            )

    def analyze_dataset(self):
        self.export_dataset_describe()
        print("dataset description created")
        self.export_p_value_calculation()
        print("linear model p-values created")
        self.export_corr_heatmap()
        print("correlation heatmap created")
        self.export_scatter()
        print("dataset scatterplot created")
        self.export_most_correlated_scatter()
        print("scatterplot of targets and their most correlated features created")
