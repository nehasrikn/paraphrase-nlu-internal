from mturk.mturk_qualification_parameters import QUALIFICATIONS_LAX

TASK_PARAMETERS = {
    'AutoApprovalDelayInSeconds': 3600*48, # hours allotted (3600 seconds/hr * n hr)
    'AssignmentDurationInSeconds': 60*25, # minutes allotted (60 seconds/min * n min)
    'Reward': '0.21',
    'Title': 'Paraphrase Simple Scenarios!',
    'Keywords': 'text, paraphrase',
    'Description': """Write 3 paraphrases of a simple sentence in a scenario.""",
    'QualificationRequirements': QUALIFICATIONS_LAX,
}