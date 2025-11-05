# vector-db

Toy vector database

## quickstart

```
cargo build --release
./target/release/vector-db bench -d fashion-mnist -s 1000
```

## todo

- [x] insert vector
- [x] dot product distance
- [x] euclidean  distance
- [x] benchmark mode
- [x] inverted file index
- [x] embedder for data insert
- [ ] namespaces
- [ ] interactive cli (readline)
- [ ] HSNW index
- [ ] index visualization
- [ ] persistence (saving database/index to disk)
- [ ] reduce inverted file index memory overhead
- [ ] python bindings


## perf changelog

### 5 nov 25
```
dataset: fashion-mnist-784-euclidean
throughput: 10K queries @ 21.74 seconds, 460 QPS
accuracy: recal@1 99.01%, recal@10 98.66%
```

> [!WARNING]
> Benchmarking method updated, results below are now invalid

### 28 oct 25 (stale)
```
dataset: fashion-mnist-784-euclidean
throughput: 10K queries @ 128.61 seconds, 78 QPS
accuracy: recal@1 99.07%, recal@10 98.64%
```

###  14 oct 25 (stale)
```
dataset: fashion-mnist-784-euclidean
throughput: 10K queries @ 323.16 seconds, 31 QPS
accuracy: recal@1 100%, recal@10 100%
```
