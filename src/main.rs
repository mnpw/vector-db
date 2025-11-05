use std::fs;
use std::path::Path;

use vector_db::{DB, Distance};

use anyhow::Error;
use clap::{Parser, Subcommand, ValueEnum};
use hdf5::File as Hdf5File;
use ndarray::Array2;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(short, long)]
    namespace: Option<String>,
    #[arg(short, long, value_enum)]
    index: Option<CliIndex>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone, Debug)]
enum CliIndex {
    Ivf,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Bench {
        #[arg(short, long, value_enum)]
        dataset: BenchDataset,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum BenchDataset {
    /// Fashion-MNIST, 784 dim, Euclidean
    FashionMnist,
}

impl BenchDataset {
    fn path(&self) -> &'static str {
        match self {
            BenchDataset::FashionMnist => "fashion-mnist-784-euclidean.hdf5",
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench { dataset, .. } => {
            if let Err(e) = run_benchmark(dataset.path()) {
                println!("Benchmark failed to run: {e:?}")
            }
        }
    };
}

fn download_dataset(dataset_name: &str, download_path: &str) -> Result<(), Error> {
    let url = format!("http://ann-benchmarks.com/{}", dataset_name);
    let response = reqwest::blocking::get(url)?;
    let bytes = response.bytes()?;
    fs::write(download_path, bytes)?;

    Ok(())
}

fn run_benchmark(dataset_path: &str) -> Result<(), Error> {
    println!("Running Benchmark");
    println!("========================\n");

    // Download dataset if needed
    if !Path::new(dataset_path).exists() {
        println!("Downloading {} dataset...", dataset_path);
        download_dataset(dataset_path, dataset_path).expect("Failed to download dataset");
        println!("Download complete!\n");
    } else {
        println!("Dataset already exists, skipping download.\n");
    }

    println!("Loading dataset from HDF5 file...");
    let file = Hdf5File::open(dataset_path)?;

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

    // Run queries and measure performance
    println!("Running benchmark queries...");
    let k = 10; // Number of neighbors to retrieve

    // Store results for later recall calculation
    let mut query_results = Vec::with_capacity(n_test);

    // Time only the actual queries
    let start = std::time::Instant::now();
    for i in 0..n_test {
        println!("Test query: [{i}]");
        let query: Vec<f64> = test_data.row(i).iter().map(|&x| x as f64).collect();
        let results = db.search(&query, k);
        query_results.push(results);
    }
    let elapsed = start.elapsed();

    // Calculate performance metrics
    let qps = n_test as f64 / elapsed.as_secs_f64();
    let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / n_test as f64;

    println!("\nCalculating recall metrics...");

    // Calculate recall metrics separately (not timed)
    let mut correct_at_1 = 0;
    let mut correct_at_10 = 0;

    for (i, results) in query_results.iter().enumerate() {
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
        for result in results {
            for (train_idx, train_vec) in db.inner.iter().enumerate() {
                if train_vec == *result && gt_neighbors.contains(&(train_idx as i32)) {
                    matches += 1;
                    break;
                }
            }
        }
        correct_at_10 += matches;
    }

    // Print results
    println!("\n========== Benchmark Results ==========");

    println!("\nPerformance Metrics (query execution only):");
    println!("--------------------------------------------");
    println!("  Total time:    {:.2} seconds", elapsed.as_secs_f64());
    println!("  Queries/sec:   {:.0} QPS", qps);
    println!("  Avg latency:   {:.2} ms/query", avg_latency_ms);

    println!("\nAccuracy Metrics:");
    println!("-----------------");
    println!(
        "  Recall@1:      {:.2}%",
        (correct_at_1 as f64 / n_test as f64) * 100.0
    );
    println!(
        "  Recall@10:     {:.2}%",
        (correct_at_10 as f64 / (n_test * k) as f64) * 100.0
    );

    println!("\n========================================");

    Ok(())
}
