$nodes = @("3.84.130.179", "3.82.51.231", "98.84.97.209", "34.224.89.142")
$key = "c:\Users\akash\Downloads\Projects\ML_Parallel_Wine\spark-key.pem"

foreach ($node in $nodes) {
    Write-Host "--- Setting up $node ---"
    scp -i $key -o StrictHostKeyChecking=no setup_spark.sh "ubuntu@$node`:/home/ubuntu/"
    ssh -i $key -o StrictHostKeyChecking=no ubuntu@$node "bash /home/ubuntu/setup_spark.sh"
}
