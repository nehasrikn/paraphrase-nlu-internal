from mturk.mturk_qualification_parameters import QUALIFICATIONS_LAX, QUALIFICATIONS_EXTRA_STRICT

TASK_PARAMETERS = {
    'AutoApprovalDelayInSeconds': 3600*48, # hours allotted (3600 seconds/hr * n hr)
    'AssignmentDurationInSeconds': 60*30, # minutes allotted (60 seconds/min * n min)
    'Reward': '0.41',
    'Title': 'Paraphrase Simple Stories!',
    'Keywords': 'text, paraphrase',
    'Description': """Write paraphrases of short and simple stories.""",
    'QualificationRequirements': QUALIFICATIONS_EXTRA_STRICT,
}