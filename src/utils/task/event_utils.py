"""
@Desc:
@Reference:
@Notes:
"""

import sys
from pathlib import Path

from collections import Counter
from typing import List

from preprocessing.event_trigger.event_ontology import EventGraph
from preprocessing.event_trigger.event_extractor import EventExtractor


def line_to_event_list(line: str):
    clean_line = line.replace(EventGraph.event_s, "").replace(EventGraph.event_e, "").strip()
    events = [one.strip() for one in clean_line.split(EventGraph.event_sep)]
    return events


def remove_empty_event_lines(src_lines: List[str], tgt_lines: List[str]):
    new_src = []
    new_tgt = []
    for s_line, t_line in zip(src_lines, tgt_lines):
        if len(line_to_event_list(s_line)) > 0:
            new_src.append(s_line)
            new_tgt.append(t_line)
    return new_src, new_tgt


if __name__ == '__main__':
    pass
