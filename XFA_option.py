import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloProFirm:
    DEFAULT_PARAMS = {
        "db_path": "trades.db",
        "initial_capital": 0.0,
        "max_drawdown": 4500.0,
        "threshold_gain": 200.0,
        "required_days": 5,
        "payout_ratio": 0.5,
        "num_days": 60,
        "num_simulations": 10000
    }
    def __init__(self, **kwargs):
        """
        オプション価値を計算したい口座ルールに合わせて初期化。

        :param db_path: トレード履歴DB
        :param initial_capital: 口座初期残高
        :param max_drawdown: ATHからの最大ドローダウン許容幅
        :param threshold_gain: initial_capital + threshold_gain を超えたら日数カウント
        :param required_days: 上記閾値を超えた日が累計何日でオプション行使
        :param payout_ratio: 行使時に受け取れる残高の割合
        :param num_days: シミュレーションする最大営業日数
        :param num_simulations: モンテカルロ試行回数
        """
        params = self.DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        self.db_path = params["db_path"]
        self.initial_capital = params["initial_capital"]
        self.max_drawdown = params["max_drawdown"]
        self.threshold_gain = params["threshold_gain"]
        self.required_days = params["required_days"]
        self.payout_ratio = params["payout_ratio"]
        self.num_days = params["num_days"]
        self.num_simulations = params["num_simulations"]

        self.df_trades = None
        self.equity_curves = []

    def fetch_trades(self) -> pd.DataFrame:
        if self.df_trades is not None:
            return self.df_trades

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time", conn)

        df["pnl_per_lot"] = df["pnl"] / df["size"]
        self.df_trades = df
        return df

    def run_option_pricing_simulation(
            self,
            lot: int,
            plot_simulations: bool = False,
            max_plot_lines: int = 100,
            daily_profit_cap: float = None
        ) -> tuple:
            df = self.fetch_trades()
            pnl_array = (df["pnl_per_lot"] * lot).values
            payoffs = np.zeros(self.num_simulations)
            temp_equity_curves = []
            temp_failed_flags = []

            for sim_idx in range(self.num_simulations):
                capital = self.initial_capital
                peak_capital = capital
                total_above_days = 0
                payoff = 0.0
                failed = False

                daily_capitals = [capital]
                success_found = False

                for day in range(self.num_days):
                    raw_pnl = np.random.choice(pnl_array)
                    if daily_profit_cap is not None:
                        raw_pnl = min(raw_pnl, daily_profit_cap)

                    capital += raw_pnl
                    if capital > peak_capital:
                        peak_capital = capital

                    if capital < peak_capital - self.max_drawdown:
                        payoff = 0.0
                        failed = True  # DDで失敗
                        daily_capitals.append(capital)
                        break

                    # 2000ドルを超えており、かつthreshold_gainも超えている場合のみカウント
                    if capital >= 2000.0 and capital >= (self.initial_capital + self.threshold_gain):
                        total_above_days += 1

                    if total_above_days >= self.required_days:
                        payoff = capital * self.payout_ratio
                        success_found = True
                        daily_capitals.append(capital)
                        break

                    daily_capitals.append(capital)

                if not success_found:
                    failed = True 

                payoffs[sim_idx] = payoff

                if plot_simulations and sim_idx < max_plot_lines:
                    temp_equity_curves.append(daily_capitals)
                    temp_failed_flags.append(failed)

            mean_payoff = payoffs.mean()
            success_prob = np.mean(payoffs > 0.0)

            if plot_simulations:
                self.equity_curves.append((lot, temp_equity_curves, temp_failed_flags))

            return (mean_payoff, success_prob)

    def plot_all_equity_curves(self):
        """
        self.equity_curves に保存された各ロット・各試行のエクイティカーブをまとめて可視化
        """
        plt.figure(figsize=(10, 6))
        color_map = plt.colormaps['tab10']  # 非推奨警告を修正

        for idx, (lot, curves, failed_flags) in enumerate(self.equity_curves):
            success_color = color_map(idx % 10)
            
            # 成功・失敗の試行を別々に描画
            for curve, failed in zip(curves, failed_flags):
                if failed:
                    plt.plot(curve, alpha=0.3, color='red')  # 失敗は赤
                else:
                    plt.plot(curve, alpha=0.3, color=success_color)  # 成功は通常色
            
            # 凡例用のサンプル曲線
            if len(curves) > 0:
                plt.plot([], [], alpha=0.8, color=success_color, label=f"Lot {lot} (success)")
                plt.plot([], [], alpha=0.8, color='red', label=f"Lot {lot} (failed)")

        plt.title("Monte Carlo Equity Curves (Option-Style)")
        plt.xlabel("Day")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_lot_analysis(self, results):
        """
        ロットサイズごとのオプション価値と成功確率の推移を描画
        
        :param results: List of tuples (lot, mean_payoff, success_prob)
        """
        lots, payoffs, probs = zip(*results)
        
        # 2つのy軸を持つグラフを作成
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左y軸 (Option Value)
        color1 = '#1f77b4'  # 青色
        ax1.set_xlabel('Lot Size')
        ax1.set_ylabel('Option Value ($)', color=color1)
        line1 = ax1.plot(lots, payoffs, color=color1, marker='o', label='Option Value')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 右y軸 (Success Probability)
        ax2 = ax1.twinx()
        color2 = '#2ca02c'  # 緑色
        ax2.set_ylabel('Success Probability (%)', color=color2)
        line2 = ax2.plot(lots, [p * 100 for p in probs], color=color2, marker='s', label='Success Probability')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 凡例
        lines1 = line1 + line2
        labels1 = [l.get_label() for l in lines1]
        ax1.legend(lines1, labels1, loc='upper center')
        
        plt.title('Option Value and Success Probability by Lot Size')
        plt.grid(True)
        plt.show()

def main():
    simulator = MonteCarloProFirm()

    lots = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []

    # 1) 各ロットについて、平均ペイオフ＆成功確率を計算
    for lot in lots:
        mean_payoff, success_prob = simulator.run_option_pricing_simulation(
            lot=lot,
            plot_simulations=False,
            max_plot_lines=50,
            daily_profit_cap=None
        )
        results.append((lot, mean_payoff, success_prob))
        print(f"[Lot={lot}] Option Value = ${mean_payoff:,.2f}, Success Prob = {success_prob*100:.2f}%")

    # 2) 結果の可視化
    simulator.plot_lot_analysis(results)

    # 3) 成功確率が最も高いロットを探す
    best_lot = None
    best_prob = -1.0
    for lot, mp, sp in results:
        if sp > best_prob:
            best_prob = sp
            best_lot = lot

    print("\n-- 結果 --")
    print(f"最も成功確率が高いロット = {best_lot}, その成功確率 = {best_prob*100:.2f}%")

    # 4) 最適ロットのエクイティカーブ表示
    mean_payoff, success_prob = simulator.run_option_pricing_simulation(
        lot=best_lot,
        plot_simulations=True,
        max_plot_lines=50,
        daily_profit_cap=None
    )
    simulator.plot_all_equity_curves()


if __name__ == "__main__":
    main()
