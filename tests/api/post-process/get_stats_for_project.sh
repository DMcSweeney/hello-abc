curl --request POST\
    --header "Content-Type: application/json"\
    --data '{"project": "Alex_Bladder", "format": "metric"}'\
    http://10.127.4.12:5001/api/post_process/get_stats_for_project