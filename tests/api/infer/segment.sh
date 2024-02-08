curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "foo", "vertebra": "L3", 
	 "series_uuid": "bar", "num_slices": "1"}'\
    http://localhost:5001/api/infer/segment
