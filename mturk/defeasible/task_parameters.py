from mturk.mturk_qualification_parameters import (
    QUALIFICATIONS_EXTRA_STRICT, 
    QUALIFICATIONS_LAX,
    QUALIFICATIONS_STRICT
)

TASK_PARAMETERS = {
    'AutoApprovalDelayInSeconds': 3600*48, # hours allotted (3600 seconds/hr * n hr)
    'AssignmentDurationInSeconds': 60*45, # minutes allotted (60 seconds/min * n min)
    'Reward': '0.21',
    'Title': 'Paraphrase Simple Sentences!',
    'Keywords': 'text, paraphrase',
    'Description': """Write 3 paraphrases of a short and simple sentence.""",
    'QualificationRequirements': QUALIFICATIONS_EXTRA_STRICT,
}