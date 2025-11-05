use std::fs;
use std::path::Path;

use vector_db::{DB, Distance};

use anyhow::Error;
use fastembed::{TextEmbedding, TextInitOptions};
use rand::seq::IndexedRandom;

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
    println!("Inserted {} vectors into database", n_train);

    // Build the IVF index
    println!("Building IVF index: {}", db.index);
    db.build_index();
    println!("Index built!\n");

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
