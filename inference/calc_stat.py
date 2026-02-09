#!/usr/bin/env python

from typing import List
from rich.console import Console
from rich.text import Text
import statistics
import fire


def compute_mean_std(numbers: List[float]) -> None:
    """
    Computes and displays the mean and standard deviation of a list of numbers.

    Args:
        numbers (List[float]): List of numerical values.

    Returns:
        None: Outputs results using rich.console.
    """
    numbers = [float(num) for num in numbers.split()]
    if not numbers:
        print("The list is empty. Please provide a list of numbers.")
        return

    mean = statistics.mean(numbers)
    std_dev = statistics.stdev(numbers) if len(numbers) > 1 else 0.0

    console = Console()

    mean_text = Text(f"Mean: {mean:.2f}", style="bold green")
    std_dev_text = Text(f"Standard Deviation: {std_dev:.2f}", style="bold blue")

    console.print(mean_text)
    console.print(std_dev_text)


if __name__ == "__main__":
    fire.Fire(compute_mean_std)
