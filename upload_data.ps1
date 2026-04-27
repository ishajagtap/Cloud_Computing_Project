$nodes = @("3.84.130.179", "3.82.51.231", "98.84.97.209", "34.224.89.142")
$key = "c:\Users\akash\Downloads\Projects\ML_Parallel_Wine\spark-key.pem"

foreach ($node in $nodes) {
    Write-Host "--- Uploading to $node ---"
    scp -i $key -o StrictHostKeyChecking=no datasets/TrainingDataset.csv datasets/ValidationDataset.csv "ubuntu@$node`:/home/ubuntu/"
}
