# Run server

```
cargo run --release -- --load-centroids=../test/centroids.json
```

# Run client

```
cargo run --bin client
```

# Benchmark 

```
ghz --proto=protos/centroids.proto --call=ml.ClusteringService.Predict --insecure --data-file=./test/observations.json localhost:8000
```

# Generate samples

```
cargo run --bin gentestdata -- --features=2 --samples=10000
```
OR
```
cargo run --bin gentestdata -- batch --batches=1000 --samples=10
```

# Generate centroids

```
cargo run --bin gencentroiddata -- --features=2 --centroids=100 --output=../test/centroids.json
```