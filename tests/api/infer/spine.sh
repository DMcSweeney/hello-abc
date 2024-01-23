curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/test", "project": "testing"}'\
    http://localhost:5001/api/infer/spine
