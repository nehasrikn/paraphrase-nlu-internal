import boto3
import botocore
import json
from datetime import datetime

from dataclasses import dataclass, asdict
import pickle
import xmltodict
import argparse
from typing import List, Optional, Dict, Union

from mturk.mturk_qualification_parameters import (
    QUALIFICATIONS_STRICT, 
    QUALIFICATIONS_LAX, 
    SANDBOX_QUALIFICATIONS
)

@dataclass
class MTurkHITData:
    task_html: str
    internal_id: str
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
    posted_hits: Optional[List[Dict]] = None # gets added afterwards

    def write_to_json(self, file_path: str) -> None:
        with open(file_path, 'w') as outfile:
            outfile.write(json.dumps(asdict(self), indent=1, sort_keys=True))

    def store_posted_batch_data(self, posted_hits: List[Dict]) -> None:
        save_keys = [
            'HITId', 'HITTypeId', 'HITStatus', 'MaxAssignments', 
            'Reward', 'AutoApprovalDelayInSeconds', 
            'AssignmentDurationInSeconds', 'RequesterAnnotation', 
            'QualificationRequirements', 'HITReviewStatus', 
            'NumberOfAssignmentsPending', 'NumberOfAssignmentsAvailable', 'NumberOfAssignmentsCompleted'
        ]
        self.posted_hits = [{k: v for k,v in h['HIT'].items() if k in save_keys} for h in posted_hits]


class MTurkHITCreator:
    
    MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

    def __init__(self, 
        aws_access_key: str,
        aws_secret_access_key: str,
        live_marketplace: bool
    ):
        self.mturk = self.connect_to_turk(aws_access_key, aws_secret_access_key, live_marketplace)
        self.check_balance()

    def check_balance(self) -> str:
        available_balance = self.mturk.get_account_balance()['AvailableBalance']
        print("I have $" + available_balance + " in my account.")
        return available_balance

    @staticmethod
    def connect_to_turk(aws_access_key: str, aws_secret_access_key: str, live_marketplace: bool):
        print('Connecting to Mechanical Turk [Live Marketplace: %s]...' % str(live_marketplace))

        if live_marketplace:
            return boto3.client(
                'mturk', 
                aws_access_key_id = aws_access_key, 
                aws_secret_access_key = aws_secret_access_key,
                region_name='us-east-1',
            )
        else:
            return boto3.client(
                'mturk',
                aws_access_key_id = aws_access_key,
                aws_secret_access_key = aws_secret_access_key,
                region_name='us-east-1',
                endpoint_url = MTurkHITCreator.MTURK_SANDBOX
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

    def create_HIT(self, task_html: str, internal_id: str, hit_type_id: str, max_assignments=3):
        HIT_metadata = {
            'HITTypeId': hit_type_id,
            'MaxAssignments': max_assignments,
            'LifetimeInSeconds': 518400,
            'Question': self._format_task_xml(task_html),
            'RequesterAnnotation': internal_id,
        }
        new_hit = self.mturk.create_hit_with_hit_type(**HIT_metadata)
        return new_hit

    def create_HITs_from_mturk_batch(self, mturk_batch: MTurkBatch) -> List[Dict]:
        posted_hits = []
        for task in mturk_batch.tasks:
            new_hit = self.create_HIT(
                task.task_html, 
                task.internal_id, 
                mturk_batch.hit_type_id, 
                max_assignments=mturk_batch.max_assignments
            )
            posted_hits.append(new_hit)

        print("Posted %d HITs!" % len(mturk_batch.tasks))
        return posted_hits

    def delete_all_active_HITs(self, HITTypeId: str) -> None:
        print('Deleting HITs of HITTypeId: %s' % HITTypeId)
        for item in self.mturk.list_hits(MaxResults=100)['HITs']:
            if item['HITTypeId'] == HITTypeId:
                self.mturk.update_expiration_for_hit(HITId=item['HITId'], ExpireAt=datetime.now())
                try:
                    self.mturk.delete_hit(HITId=item['HITId'])
                    print('Successfully deleted HIT %s' % item['HITId'])
                except Exception as e:
                    print('Could not delete HIT %s' % item['HITId'])
                    continue

    def trigger_email_notifications(self, HITTypeId: str, email: str) -> None:
        print('Turning email notifications on for HITType: %s' % HITTypeId)
        notification_response = self.mturk.update_notification_settings(
            HITTypeId=HITTypeId,
            Notification={
                'Destination': email,
                'Transport': 'Email',
                'Version': '2006-05-05',
                'EventTypes': [
                    'AssignmentSubmitted',
                ]
            },
        )
        print(notification_response)

    def get_assignment(self, AssignmentId: str):
        print(self.mturk.get_assignment(AssignmentId=AssignmentId)['Assignment']['Answer'])


