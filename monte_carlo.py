import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloProFirm:
    DEFAULT_PARAMS = {
        "db_path": "trades.db",
        "initial_capital": 50_000.0,
        "profit_target": 3_000.0,
        "max_drawdown": 2_000.0,
        "num_days": 20,
        "num_simulations": 10_000
    }

    def __init__(self, **kwargs):
        """
        :param kwargs: DEFAULT_PARAMSの値を上書きする任意のパラメータ
        """
        params = self.DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        self.db_path = params["db_path"]
        self.initial_capital = params["initial_capital"]
        self.profit_target = params["profit_target"]
        self.max_drawdown = params["max_drawdown"]
        self.num_days = params["num_days"]
        self.num_simulations = params["num_simulations"]

        self.df_trades = None
        self.equity_curves = []
        self.success_curves = []  # 成功したケースの曲線
        self.failure_curves = []  # 失敗したケースの曲線

    def fetch_trades(self) -> pd.DataFrame:
        if self.df_trades is not None:
            return self.df_trades

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time", conn)

        df["pnl_per_lot"] = df["pnl"] / df["size"]
        self.df_trades = df
        return df

    def run_simulation_for_lot(self, lot: int, plot_simulations: bool = False, max_plot_lines: int = 50) -> float:
        df = self.fetch_trades()
        pnl_array = (df["pnl_per_lot"] * lot).values

        success_count = 0
        temp_success_curves = []
        temp_failure_curves = []

        for sim_idx in range(self.num_simulations):
            capital = self.initial_capital
            peak_capital = capital
            is_success = False
            daily_capitals = [capital]

            for day in range(self.num_days):
                raw_pnl = np.random.choice(pnl_array)
                daily_pnl = min(raw_pnl, 1499.0)
                capital += daily_pnl
                
                if capital > peak_capital:
                    peak_capital = capital

                dd_limit = peak_capital - self.max_drawdown

                if capital < dd_limit:
                    daily_capitals.append(capital)
                    break

                if (capital - self.initial_capital) >= self.profit_target:
                    is_success = True
                    daily_capitals.append(capital)
                    break

                daily_capitals.append(capital)

            if is_success:
                success_count += 1
                if plot_simulations and len(temp_success_curves) < max_plot_lines:
                    temp_success_curves.append(daily_capitals)
            elif plot_simulations and len(temp_failure_curves) < max_plot_lines:
                temp_failure_curves.append(daily_capitals)

        success_prob = success_count / self.num_simulations

        if plot_simulations:
            self.success_curves.append((lot, temp_success_curves))
            self.failure_curves.append((lot, temp_failure_curves))

        return success_prob

    def find_best_lot_in_range(self, lot_min: int = 5, lot_max: int = 10) -> tuple:
        best_lot = None
        best_prob = -1.0

        for lot in range(lot_min, lot_max + 1):
            prob = self.run_simulation_for_lot(lot=lot, plot_simulations=False)
            if prob > best_prob:
                best_prob = prob
                best_lot = lot

        return best_lot, best_prob

    def plot_all_equity_curves(self):
        plt.figure(figsize=(10, 6))
        color_map = plt.colormaps['tab10']

        for idx, (lot, success_curves) in enumerate(self.success_curves):
            color = color_map(idx % 10)
            for curve in success_curves:
                plt.plot(curve, alpha=0.3, color=color)
            if success_curves:
                plt.plot(success_curves[0], alpha=0.8, color=color, 
                        label=f"Lot {lot} (success)")

        for idx, (lot, failure_curves) in enumerate(self.failure_curves):
            for curve in failure_curves:
                plt.plot(curve, alpha=0.3, color='red')
            if failure_curves:
                plt.plot(failure_curves[0], alpha=0.8, color='red', 
                        label=f"Lot {lot} (failure)")

        plt.title("Monte Carlo Equity Curves (Various Lots)")
        plt.xlabel("Day")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    simulator = MonteCarloProFirm()  # デフォルトパラメータを使用
    
    best_lot, best_prob = simulator.find_best_lot_in_range(lot_min=5, lot_max=10)
    print(f"【ロット範囲 5～10 の中で最も成功確率が高いロット】")
    print(f"  Lot = {best_lot},  成功確率 = {best_prob*100:.2f}%")

    simulator.run_simulation_for_lot(lot=best_lot, plot_simulations=True, max_plot_lines=50)
    simulator.plot_all_equity_curves()

if __name__ == "__main__":
    main()