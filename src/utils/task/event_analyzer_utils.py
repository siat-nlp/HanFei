import os
import sys
from multiprocessing import Process

import spacy
from typing import List
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter

from src.configuration.constants import BASE_DIR
from preprocessing.event_trigger.event_predictor import EventPredictor


class EventAnalyzer(object):
    def event_trainings_nums(self, event_predictor: EventPredictor):
        event_graph = event_predictor.event_graph
        counter = Counter()
        for id, event in event_graph.events.items():
            counter[id] = len(event.extracted_sents)
        return counter

    def analyze_events(self, event_predictor: EventPredictor):
        event_graph = event_predictor.event_graph
        while True:
            event_string = input("输入event_string")
            if event_string in ["quit", "q"]:
                break
            event = event_graph.find_event(event_string)
            if not event:
                print(f"no {event_string}")
                continue
            else:
                print(f"{event_string} next candidates: {event_graph.next_candidates(event.uuid, limit=3)}")
                print(f"{event_string} prev candidates: {event_graph.prev_events(event.uuid, limit=3)}")
