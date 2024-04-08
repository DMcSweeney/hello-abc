# curl --request POST\
# 	 --header "Content-Type: application/json"\
# 	 --data '{"input_path": "/data/inputs/Denis/dataset148_test_set/171140644_pCT.nii", "study_uuid": "1",
#       "project": "denisTesting", "patient_id": "171140644", "series_uuid": "171140644", "acquisition_date": "None"}'\
#     http://10.127.4.12:5001/api/infer/spine


# curl --request POST\
# 	 --header "Content-Type: application/json"\
# 	 --data '{"input_path": "/data/inputs/Denis/dataset148_test_set/300035172_pCT.nii", "study_uuid": "1",
#       "project": "denisTesting", "patient_id": "300035172", "series_uuid": "300035172", "acquisition_date": "None"}'\
#     http://10.127.4.12:5001/api/infer/spine

# curl --request POST\
# 	 --header "Content-Type: application/json"\
# 	 --data '{"input_path": "/data/inputs/Donal/oesophagus_pCT/data/additional_nii/127469153.nii", "study_uuid": "1",
#       "project": "denisTesting", "patient_id": "testDICOM", "series_uuid": "testDICOM", "acquisition_date": "None"}'\
#     http://10.127.4.12:5001/api/infer/spine

curl --request POST\
	 --header "Content-Type: application/json"\
	 --data '{"input_path": "/data/inputs/ABC-Toolkit/106200039", "study_uuid": "1",
      "project": "denisTesting", "patient_id": "DICOM", "series_uuid": "106200039", "acquisition_date": "None"}'\
    http://10.127.4.12:5001/api/infer/spine