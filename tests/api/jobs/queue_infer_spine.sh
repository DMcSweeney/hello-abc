curl -X POST\
    --header "Content-Type: application/json"\
    --data '{"input_path": "/data/inputs/test", "project": "Alex_Bladder"}'\
    http://localhost:5001/api/jobs/infer/spine