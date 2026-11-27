#!/usr/bin/env python
"""Plot per-channel ripple candidate rates and mouse-level area summaries."""

import argparse
from pathlib import Path

import pandas as pd

from q3_detect_ripples import AREAS, RESULTS_DIR
from q3_plot_sharp_wave_rates import mouse_area_means, mouse_area_peaks, plot_area_summary, plot_mouse


RATE = "ripples_per_min"


def run(results_dir):
    rates = pd.read_csv(results_dir / "q3_ripple_channel_rates.csv")
    output = results_dir / "rate_plots"
    output.mkdir(parents=True, exist_ok=True)
    for mouse_id, mouse in rates.groupby("mouse_id", sort=True):
        if set(mouse.brain_area) != set(AREAS):
            raise ValueError(f"Missing area data for mouse {mouse_id}")
        plot_mouse(mouse_id, mouse, output, rate=RATE, position="channel_row",
                   event_label="Ripple", file_stem="ripple")
    plot_area_summary(mouse_area_means(rates, rate=RATE),
                      results_dir / "q3_ripple_rate_area_summary.png",
                      title="Ripple candidate rates by area",
                      ylabel="Mean candidates/min across channels", rate=RATE)
    plot_area_summary(mouse_area_peaks(rates, rate=RATE, position="channel_row"),
                      results_dir / "q3_ripple_rate_peak_channel_summary.png",
                      title="Peak-channel ripple rates by area",
                      ylabel="Candidates/min at highest-count channel", rate=RATE)
    print(f"Saved {rates.mouse_id.nunique()} channel profiles and two area summaries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    run(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
