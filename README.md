# vector-db

Toy vector database

## TODO

- [x] insert vector
- [x] dot product distance
- [x] euclidean  distance
- [x] benchmark mode
- [x] inverted file index
- [ ] saving + restoring inverted file index
- [ ] reduce inverted file index memory overhead

## Perf changelog

## 28 oct
dataset: fashion-mnist-784-euclidean
throughput: 10K queries @ 128.61 seconds, 78 QPS
accuracy: recal@1 99.07%, recal@10 98.64%

##  14 oct
dataset: fashion-mnist-784-euclidean
throughput: 10K queries @ 323.16 seconds, 31 QPS
accuracy: recal@1 100%, recal@10 100%
