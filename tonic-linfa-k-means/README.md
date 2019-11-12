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

# Generate centroids

```
cargo run --bin gencentroiddata -- --features=2 --centroids=100
```