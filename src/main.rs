use std::fs;
use std::path::Path;

use anyhow::Error;

fn main() {
    println!("Fashion-MNIST Benchmark");
    println!("========================\n");

    // Download dataset if needed
    let dataset = "fashion-mnist-784-euclidean.hdf5";
    if !Path::new(dataset).exists() {
        println!("Downloading Fashion-MNIST dataset...");
        download_dataset(dataset, dataset).expect("Failed to download dataset");
        println!("Download complete!\n");
    } else {
        println!("Dataset already exists, skipping download.\n");
    }

    // Run benchmark
    run_benchmark(dataset).expect("Benchmark failed");
}

fn download_dataset(dataset_name: &str, download_path: &str) -> Result<(), Error> {
    let url = format!("http://ann-benchmarks.com/{}", dataset_name);
    let response = reqwest::blocking::get(url)?;
    let bytes = response.bytes()?;
    fs::write(download_path, bytes)?;

    Ok(())
}

fn run_benchmark(dataset_path: &str) -> Result<(), Error> {
    use hdf5::File;
    use ndarray::Array2;

    println!("Loading dataset from HDF5 file...");
    let file = File::open(dataset_path)?;

    // Read training vectors
    let train_dataset = file.dataset("train")?;
    let train_data: Array2<f32> = train_dataset.read_2d()?;
    let (n_train, dim) = train_data.dim();
    println!("Training vectors: {} x {}", n_train, dim);

    // Read test vectors
    let test_dataset = file.dataset("test")?;
    let test_data: Array2<f32> = test_dataset.read_2d()?;
    let (n_test, _) = test_data.dim();
    println!("Test vectors: {} x {}", n_test, dim);

    // Read ground truth neighbors
    let neighbors_dataset = file.dataset("neighbors")?;
    let neighbors: Array2<i32> = neighbors_dataset.read_2d()?;
    let (_, k_neighbors) = neighbors.dim();
    println!("Ground truth neighbors: {} per query\n", k_neighbors);

    // Create database with Euclidean distance
    println!("Building database...");
    let mut db = DB::new(dim, Distance::Euclidean);

    // Insert all training vectors
    for i in 0..n_train {
        let vector: Vec<f64> = train_data.row(i).iter().map(|&x| x as f64).collect();
        db.insert(vector);
    }
    println!("Inserted {} vectors into database\n", n_train);

    // Run queries and calculate metrics
    println!("Running benchmark queries...");
    let k = 10; // Number of neighbors to retrieve
    let start = std::time::Instant::now();

    let mut correct_at_1 = 0;
    let mut correct_at_10 = 0;

    for i in 0..n_test {
        println!("Test query: [{i}]");
        let query: Vec<f64> = test_data.row(i).iter().map(|&x| x as f64).collect();
        let results = db.search(&query, k);

        // Get ground truth for this query
        let gt_neighbors: Vec<i32> = neighbors.row(i).iter().take(k).copied().collect();

        // Check recall@1
        if let Some(first_result) = results.first() {
            // Find the index of the first result in the training data
            for (train_idx, train_vec) in db.inner.iter().enumerate() {
                if train_vec == *first_result && gt_neighbors[0] == train_idx as i32 {
                    correct_at_1 += 1;
                    break;
                }
            }
        }

        // Check recall@10
        let mut matches = 0;
        for result in &results {
            for (train_idx, train_vec) in db.inner.iter().enumerate() {
                if train_vec == *result && gt_neighbors.contains(&(train_idx as i32)) {
                    matches += 1;
                    break;
                }
            }
        }
        correct_at_10 += matches;
    }

    let elapsed = start.elapsed();
    let qps = n_test as f64 / elapsed.as_secs_f64();
    let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / n_test as f64;

    // Print results
    println!("\nResults");
    println!("=======");
    println!(
        "Recall@1:   {:.2}%",
        (correct_at_1 as f64 / n_test as f64) * 100.0
    );
    println!(
        "Recall@10:  {:.2}%",
        (correct_at_10 as f64 / (n_test * k) as f64) * 100.0
    );
    println!("\nPerformance:");
    println!("- Total time: {:.2} seconds", elapsed.as_secs_f64());
    println!("- QPS: {:.0} queries/second", qps);
    println!("- Avg latency: {:.2} ms", avg_latency_ms);

    Ok(())
}

type Vector = Vec<f64>;

struct DB {
    pub inner: Vec<Vector>,
    dim: usize,
    distance_metric: Distance,
}

enum Distance {
    Dot,
    Euclidean,
}

impl Distance {
    fn compute(&self, first: &Vector, second: &Vector) -> f64 {
        assert_eq!(
            first.len(),
            second.len(),
            "vector size must match for euclidean distance"
        );

        match self {
            Distance::Dot => Self::compute_dot(first, second),
            Distance::Euclidean => Self::compute_euclidean(first, second),
        }
    }

    // poor man's dot product
    fn compute_dot(first: &Vector, second: &Vector) -> f64 {
        let mut res = 0.0;
        first
            .iter()
            .zip(second.iter())
            .for_each(|(a, b)| res = res + (a * b));

        res
    }

    // euclidean distance (L2)
    fn compute_euclidean(first: &Vector, second: &Vector) -> f64 {
        let mut sum_sq = 0.0;
        first.iter().zip(second.iter()).for_each(|(a, b)| {
            let diff = a - b;
            sum_sq += diff * diff;
        });

        sum_sq.sqrt()
    }
}

impl DB {
    fn new(dimension: usize, distance_metric: Distance) -> Self {
        Self {
            inner: vec![],
            dim: dimension,
            distance_metric,
        }
    }

    fn insert(&mut self, vector: Vector) {
        assert_eq!(vector.len(), self.dim);
        self.inner.push(vector);
    }

    fn search(&self, vector: &Vector, count: usize) -> Vec<&Vector> {
        assert_eq!(vector.len(), self.dim);
        let mut distances = Vec::with_capacity(self.inner.len());

        // 1. compute distance with every vector in self.inner
        for (id, v) in self.inner.iter().enumerate() {
            let dist = self.distance_metric.compute(v, vector);
            distances.push((id, dist));
        }

        // 2. sort distances
        match self.distance_metric {
            // Euclidean distanace = small => closer
            Distance::Euclidean => distances.sort_by(|a, b| a.1.total_cmp(&b.1)),
            // Dot distance = large => closer
            Distance::Dot => distances.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        // 3. return top `count`
        let top_idx = &distances[0..count];
        let mut top = Vec::with_capacity(top_idx.len());
        for (id, _dist) in top_idx {
            top.push(self.inner.get(*id).unwrap());
        }

        top
    }
}

#[test]
fn test_db() {
    let mut db = DB::new(4, Distance::Dot);
    db.insert(vec![0.05, 0.61, 0.76, 0.74]);
    db.insert(vec![0.19, 0.81, 0.75, 0.11]);
    db.insert(vec![0.36, 0.55, 0.47, 0.94]);
    db.insert(vec![0.18, 0.01, 0.85, 0.80]);
    db.insert(vec![0.24, 0.18, 0.22, 0.44]);
    db.insert(vec![0.35, 0.08, 0.11, 0.44]);

    let result = db.search(&vec![0.2, 0.1, 0.9, 0.7], 1);
    assert_eq!(result, vec![&vec![0.18, 0.01, 0.85, 0.80]]);
}
