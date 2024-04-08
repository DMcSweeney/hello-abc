curl -X POST\
    --header "Content-Type: application/json"\
    --data '{"input_path": "/data/inputs/test", "project": "jobTesting", "vertebra": "L3"}'\
    http://localhost:5001/api/jobs/infer/segment