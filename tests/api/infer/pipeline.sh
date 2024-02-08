curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "foo", "series_uuid": "bar"}'\
    http://localhost:5001/api/infer/spine


curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "foo", "series_uuid": "bar", "vertebra": "L3", "num_slices": "1"}'\
    http://localhost:5001/api/infer/segment
