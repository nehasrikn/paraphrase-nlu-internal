import boto3
import botocore
import json
from datetime import datetime

from dataclasses import dataclass
import pickle
import xmltodict
import argparse
from typing import List, Optional, Dict, Union

from mturk_qualification_parameters import (
    QUALIFICATIONS_STRICT, 
    QUALIFICATIONS_LAX, 
    SANDBOX_QUALIFICATIONS
)

@dataclass
class MTurkHITData:
    task_html: str
    internal_ID: str
    split: str
    example_data: Dict[str, str]

@dataclass
class MTurkHITTypeParameters:
    AutoApprovalDelayInSeconds: int
    AssignmentDurationInSeconds: int
    Reward: str
    Title: str
    Keywords: str
    Description: str
    QualificationRequirements: List[Dict]

@dataclass
class MTurkBatch:
    tasks: List[MTurkHITData]
    origin_split: str
    dataset_name: str
    example_ids: List[int]
    max_assignments: int
    post_date: str
    requestor_note: str
    hit_type_id: str

    def write_batch_to_json(self, file_path: str) -> None:
        with open(file_path, 'w') as outfile:
            outfile.write(json.dumps(asdict(self), indent=1, sort_keys=True))


class MTurkHITCreator:
    
    MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    AWS_ACCESS_KEY_ID = 'AKIA3HQJKSL4YZUFYGQ4'
    AWS_SECRET_ACCESS_KEY = '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'


    def __init__(self):
        self.mturk = self.connect_to_turk()

    def check_balance(self) -> str:
        available_balance = self.mturk.get_account_balance()['AvailableBalance']
        print("I have $" + available_balance + " in my account.")
        return available_balance

    def connect_to_turk(self):
        print('Connecting to Mechanical Turk...')
        return boto3.client(
            'mturk',
            aws_access_key_id = MTurkHITCreator.AWS_ACCESS_KEY_ID,
            aws_secret_access_key = MTurkHITCreator.AWS_SECRET_ACCESS_KEY,
            region_name='us-east-1',
            endpoint_url = MTurkHITCreator.MTURK_SANDBOX #comment out for live marketplace
        )

    def _format_task_xml(self, task_html: str) -> str:
        question_xml = """
            <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
               <HTMLContent>
                    <![CDATA[<!DOCTYPE html>""" + task_html + """]]>
               </HTMLContent>
               <FrameHeight>600</FrameHeight>
            </HTMLQuestion>"""
        return question_xml

    def create_HIT_type(self, hit_type_parameters: MTurkHITTypeParameters) -> Dict[str, str]:
        hit_type = self.mturk.create_hit_type(**asdict(hit_type_parameters))
        print("Created HITType with HITTypeId:", hit_type['HITTypeId']) 
        return hit_type

    def create_HIT(self, task_html: str, internal_ID: int, hit_type_id: str, max_assignments=3):
        HIT_metadata = {
            'HITTypeId': hit_type_id,
            'MaxAssignments': max_assignments,
            'LifetimeInSeconds': 518400,
            'Question': self._format_task_xml(task_html),
            'RequesterAnnotation': internal_ID,
        }
        new_hit = self.mturk.create_hit_with_hit_type(**HIT_metadata)
        return new_hit


    def create_HITs_from_mturk_batch(mturk_batch: MTurkBatch) -> None: #tasks: List[MTurkHITData], HIT_type_id: str, max_assignments=3) -> None:
        for task in mturk_batch['tasks']:
            new_hit = create_HIT(
                task['task_html'], 
                task['internal_id'], 
                mturk_batch['hit_type_id'], 
                max_assignments=mturk_batch['max_assignments']
            )

        print("Posted %d HITs!" % len(data))

