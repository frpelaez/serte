import matplotlib.pyplot as plt


class DecompositionResult:
    def __init__(self, observed, trend, seasonal, resid):
        self.observed = observed
        self.trend = trend
        self.seasonal = seasonal
        self.resid = resid

    def plot(self):
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

        axes[0].plot(self.observed.data, label="Observed")
        axes[0].legend(loc="upper left")
        axes[0].set_title("Decomposition of Time Series")

        axes[1].plot(self.trend.data, label="Trend", color="orange")
        axes[1].legend(loc="upper left")

        axes[2].plot(self.seasonal.data, label="Seasonality", color="green")
        axes[2].legend(loc="upper left")

        axes[3].scatter(
            self.resid.data.index, self.resid.data, label="Residuals", color="red", s=10
        )
        axes[3].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[3].legend(loc="upper left")

        plt.tight_layout()
        plt.show()
