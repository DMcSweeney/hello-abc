curl --request POST\
    --header "Content-Type: application/json"\
    --data '{"project": "testing", "format": "metric"}'\
    http://localhost:5001/api/post_process/get_stats_for_project