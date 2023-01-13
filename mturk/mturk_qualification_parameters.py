QUALIFICATIONS_STRICT = [
    {
        'QualificationTypeId': '00000000000000000071', # Worker_Locale
        'Comparator': 'EqualTo',
        'LocaleValues': [{'Country': 'US',},{'Country': 'US',}]
    },               
    {
        'QualificationTypeId': '00000000000000000040', # Worker_NumberHITsApproved  
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [500],
    },
    {
        'QualificationTypeId': '000000000000000000L0', # Worker_PercentAssignmentsApproved  
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [95],
    },
]

QUALIFICATIONS_EXTRA_STRICT = [
    {
        'QualificationTypeId': '00000000000000000071', # Worker_Locale
        'Comparator': 'In',
        'LocaleValues': [
            {'Country': 'US',}, 
            {'Country': 'CA',}, 
            {'Country': 'GB',}, 
            {'Country': 'AU',},
            {'Country': 'NZ',}
        ]
    },
    {
        'QualificationTypeId': '00000000000000000040', # Worker_NumberHITsApproved
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [500],
    },
    {
        'QualificationTypeId': '000000000000000000L0', # Worker_PercentAssignmentsApproved
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [98],
    },     
]

QUALIFICATIONS_LAX = [
    {
        'QualificationTypeId': '00000000000000000071', # Worker_Locale
        'Comparator': 'In',
        'LocaleValues': [
            {'Country': 'US',}, 
            {'Country': 'CA',}, 
            {'Country': 'GB',}, 
            {'Country': 'AU',},
            {'Country': 'NZ',}
        ]
    },
    {
        'QualificationTypeId': '00000000000000000040', # Worker_NumberHITsApproved
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [100],
    },
    {
        'QualificationTypeId': '000000000000000000L0', # Worker_PercentAssignmentsApproved
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [95],
    },     
]

SANDBOX_QUALIFICATIONS = [
    {
        'QualificationTypeId': '00000000000000000071',
        'Comparator': 'EqualTo',
        'LocaleValues': [{'Country': 'US',},]
    },
    {
        'QualificationTypeId': '000000000000000000L0',
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [95],
    },
]