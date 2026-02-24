#!/usr/bin/env python

import re
import json

from rich.console import Console

console = Console()
# path = "logs/scaling_test_time_compute.log"
# path = "logs/scaling_test_time_compute.oxford.log"
# path = "logs/scaling_test_time_compute.wiki.log"
# path = "logs/scaling_test_time_compute.slang.log"
path = "logs/scaling_test_time_compute.3d-ex.log"

text = [l.strip() for l in open(path, "r").readlines()]
for line in text:
    if "│ word-interpretation │" in line:
        console.print(line)
        continue
        scores = re.findall(r"\d+\.\d+", line)
        if len(scores) > 0:
            for i in range(len(scores)):
                scores[i] = float(scores[i])
                console.print(f"Score: {scores[i]}")
