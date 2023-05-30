"""
@Desc:
@Reference:
@Notes:
"""

import os
import sys
from pathlib import Path

from collections import Counter
from typing import List

from src.configuration.constants import BASE_DIR
from src.utils.task.event_utils import line_to_event_list
from preprocessing.event_trigger.event_extractor import EventExtractor


def data_size(file: str):
    with open(file, "r", encoding="utf-8") as fr:
        return len(fr.readlines())


def text_stat(file: str):
    with open(file, "r", encoding="utf-8") as fr:
        file_lines = fr.readlines()
        stat = Counter()
        for line in file_lines:
            sents = line.strip().split(".")
            tokens = line.strip().split()
            stat["sents"] += len(sents)
            stat["tokens"] += len(tokens)
        return stat


def event_stat(file: str):
    with open(file, "r", encoding="utf-8") as fr:
        event_lines = fr.readlines()
        stat = Counter()
        for line in event_lines:
            events = line_to_event_list(line)
            stat["events"] += len(events)
        return stat


def event_graph_stat(event_extractor: EventExtractor):
    stat = Counter()
    event_graph = event_extractor.event_graph
    stat["events"] = event_graph.nodes_num
    stat["relations"] = event_graph.edges_num
    stat["avg_degree"] = event_graph.avg_degree
    stat["triggers"] = event_graph.triggers_num
    return stat


def parse_files(src_file, tgt_file, event_file):
    counter = text_stat(src_file) + text_stat(tgt_file) + event_stat(event_file)
    # src 的 events 没算
    counter["data_size"] = data_size(src_file)
    counter["events"] += counter["data_size"]
    return counter


def parse_event_graphs(dataset_name: str):
    save_path = f"{BASE_DIR}/output/event-trigger/cache/{dataset_name}_event_graph.pkl"
    if os.path.exists(save_path):
        print(f"extractor loaded from {save_path}")
        event_extractor = EventExtractor.load(save_path)
        return event_graph_stat(event_extractor)
