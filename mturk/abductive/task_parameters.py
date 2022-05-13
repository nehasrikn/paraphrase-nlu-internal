from mturk_qualification_parameters import QUALIFICATIONS_LAX

TASK_PARAMETERS = {
    'AutoApprovalDelayInSeconds': 3600*48, # hours allotted (3600 seconds/hr * n hr)
    'AssignmentDurationInSeconds': 60*20, # minutes allotted (60 seconds/min * n min)
    'Reward': '0.41',
    'Title': 'Paraphrase Simple Stories!',
    'Keywords': 'text, paraphrase',
    'Description': """Write paraphrases of simple stories.""",
    'QualificationRequirements': QUALIFICATIONS_LAX,
}