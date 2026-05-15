"""Tiny ASCII visualization helpers — no plotting deps required."""

from __future__ import annotations

from .types import RoundReport


def price_sparkline(reports: list[RoundReport], *, width: int = 60, height: int = 10) -> str:
    """Render a simple ASCII line chart of clearing price over rounds.

    Output is a multi-line string suitable for printing directly.
    """
    if not reports:
        return "(no rounds)"

    # Include the opening price of round 0 as the starting point so the very
    # first clearing move shows up on the chart.
    prices = [reports[0].opening_price] + [r.clearing_price for r in reports]
    n = len(prices)
    width = min(width, n)
    bucket = max(1, n // width)
    sampled = [
        sum(prices[i : i + bucket]) / len(prices[i : i + bucket])
        for i in range(0, n, bucket)
    ][:width]

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo

    rows: list[str] = []
    if span < 1e-9:
        # Constant series — draw a single mid-row line.
        mid = height // 2
        for row in range(height, 0, -1):
            marker = "*" * len(sampled) if row == mid else " " * len(sampled)
            rows.append(f"{lo:7.3f} | {marker}")
    else:
        for row in range(height, 0, -1):
            threshold_lo = lo + span * (row - 1) / height
            threshold_hi = lo + span * row / height
            line_chars = []
            for p in sampled:
                line_chars.append("*" if threshold_lo <= p <= threshold_hi else " ")
            label_val = lo + span * (row - 0.5) / height
            rows.append(f"{label_val:7.3f} | " + "".join(line_chars))

    axis = " " * 9 + "+" + "-" * len(sampled)
    last_round = len(reports) - 1
    label = " " * 10 + f"round 0{' ' * max(0, len(sampled) - 14)}round {last_round}"
    return "\n".join(rows + [axis, label])


def equity_lines(reports: list[RoundReport]) -> dict[str, list[float]]:
    """Per-agent equity series for plotting or printing."""
    series: dict[str, list[float]] = {}
    for report in reports:
        for portfolio in report.portfolios:
            series.setdefault(portfolio.agent_id, []).append(
                portfolio.equity(report.clearing_price)
            )
    return series
