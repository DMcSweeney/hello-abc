curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "testing", "vertebra": "L3", "num_slices": "1"}'\
    http://localhost:5001/api/jobs/infer/segment
