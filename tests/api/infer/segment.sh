curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "testing", "vertebra": "L3", "slice_number": "48", "num_slices": "1"}'\
    http://localhost:5001/api/infer/segment
