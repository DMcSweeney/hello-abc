## Web-ABC

Repo for launching the ABC-toolkit web app
`src` -> ABC-core, most containers will import this.
 
# TODO
- Sanity check, should use multi-slice variance to highlight examples that may need checking.
- DICOM server + permanent store (e.g. planning CTs for HnN) vs. temp store (CBCTs) 
- Postgres container (store series UIDs, results [paths2preds, slice nums, areas, densities])


- Add task queues and message brokers for spine and segment (celery -> redis, RabbitMQ).
- nginx forward-proxy (low-priority)


