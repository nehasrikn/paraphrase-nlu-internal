# AWS_ACCESS_KEY_ID = 'AKIA3HQJKSL4YZUFYGQ4' # associated with nehaspsrik@gmail.com
# AWS_SECRET_ACCESS_KEY = '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'


python -m abductive.abductive_hit_creator \
	--aws_access_key='AKIA3HQJKSL4YZUFYGQ4' \
	--aws_secret_access_key='51DNsHKAT+SiThFybgaEIZS8YT1sJyHt6zsNLSHE' \
	--split='test' \
	--requestor_note='initial pilot' \
	--num_examples=115 \
	--max_assignments=3 \
	--hit_type_id='32TFGTUJLO3J751NEB1G1MT9DBKIP8' \

	# --live_marketplace \
