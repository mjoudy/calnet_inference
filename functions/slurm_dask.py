from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def setup_cluster(queue="dev_single", cores=1, memory="8GB", walltime="00:30:00", jobs=20, log_dir="./logs"):
    cluster = SLURMCluster(
        queue=queue,
        cores=cores,
        memory=memory,
        processes=1,
        walltime=walltime,
        job_extra_directives=["--ntasks=1"],
        log_directory=log_dir,
    )
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    return cluster, client
