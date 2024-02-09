curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/1055", "project": "foo", "series_uuid": "baz"}'\
    http://localhost:5001/api/infer/spine


curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/1055", "project": "foo", "series_uuid": "baz", "vertebra": "L3", "num_slices": "1"}'\
    http://localhost:5001/api/infer/segment
