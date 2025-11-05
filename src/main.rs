use std::{default, fs};

use vector_db::{DB, Distance};

use anyhow::Error;
use clap::{Parser, Subcommand, ValueEnum};
use hdf5::File as Hdf5File;
use ndarray::Array2;
use tracing::{debug, error, info};

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
        #[arg(short, long, value_enum, default_value_t=BenchDataset::FashionMnist)]
        dataset: BenchDataset,
        #[arg(short, long)]
        size: Option<usize>,
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
    tracing_subscriber::fmt().init();

    match cli.command {
        Commands::Bench { dataset, size } => {
            let benchmark_runner = BenchmarkRunner::new();
            if let Err(e) = benchmark_runner.run_benchmark(dataset.path(), size) {
                error!("Benchmark failed to run: {e:?}")
            }
        }
    };
}

struct BenchmarkRunner {
    dataset_download_base_url: String,
    dataset_download_path: String,
    k_neighbours: usize,
}

impl BenchmarkRunner {
    fn new() -> Self {
        BenchmarkRunner {
            dataset_download_base_url: "http://ann-benchmarks.com".to_string(),
            dataset_download_path: "dataset/".to_string(),
            k_neighbours: 10,
        }
    }

    fn download_dataset(&self, dataset_name: &str) -> Result<(), Error> {
        let url = format!("{}/{}", self.dataset_download_base_url, dataset_name);
        let response = reqwest::blocking::get(url)?;
        let bytes = response.bytes()?;
        fs::write(
            format!("{}/{}", self.dataset_download_path, dataset_name),
            bytes,
        )?;

        Ok(())
    }

    fn run_benchmark(&self, dataset_name: &str, test_size: Option<usize>) -> Result<(), Error> {
        info!("Running Benchmark");

        let dataset_path = format!("{}/{}", self.dataset_download_path, dataset_name);
        if !fs::exists(&self.dataset_download_path)? {
            fs::create_dir(&self.dataset_download_path)?;
        }

        if !fs::exists(&dataset_path)? {
            self.download_dataset(dataset_name)?;
            info!("Downloaded {} dataset", dataset_path);
        }

        let file = Hdf5File::open(dataset_path)?;

        // Read training vectors
        let train_dataset = file.dataset("train")?;
        let train_data: Array2<f32> = train_dataset.read_2d()?;
        let (n_train, dim) = train_data.dim();
        info!("Training vectors: {} x {}", n_train, dim);

        // Read test vectors
        let test_dataset = file.dataset("test")?;
        let test_data: Array2<f32> = test_dataset.read_2d()?;
        let (n_test, _) = test_data.dim();
        info!("Test vectors: {} x {}", n_test, dim);

        // Determine actual number of queries to run
        let n_queries = test_size.map_or(n_test, |size| size.min(n_test));
        if n_queries < n_test {
            info!("Limiting queries to {} (test_size parameter)", n_queries);
        }

        // Read ground truth neighbors
        let neighbors_dataset = file.dataset("neighbors")?;
        let neighbors: Array2<i32> = neighbors_dataset.read_2d()?;
        let (_, k_neighbors) = neighbors.dim();
        info!("Ground truth neighbors: {} per query", k_neighbors);

        // Create database with Euclidean distance
        info!("Building database...");
        let mut db = DB::new(dim, Distance::Euclidean);

        // Insert all training vectors
        for i in 0..n_train {
            let vector: Vec<f64> = train_data.row(i).iter().map(|&x| x as f64).collect();
            db.insert(vector);
        }
        info!("Inserted {} vectors into database", n_train);

        // Build the IVF index
        info!("Building IVF index: {}", db.index);
        db.build_index();
        info!("Index built!");

        // Run queries and measure performance
        info!("Running benchmark queries...");

        // Store results for later recall calculation
        let mut query_results = Vec::with_capacity(n_queries);

        // Time only the actual queries
        let start = std::time::Instant::now();
        for i in 0..n_queries {
            debug!("Test query: [{i}]");
            let query: Vec<f64> = test_data.row(i).iter().map(|&x| x as f64).collect();
            let results = db.search(&query, self.k_neighbours);
            query_results.push(results);
        }
        let elapsed = start.elapsed();

        // Calculate performance metrics
        let qps = n_queries as f64 / elapsed.as_secs_f64();
        let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / n_queries as f64;

        info!("Calculating recall metrics...");

        // Calculate recall metrics separately (not timed)
        let mut correct_at_1 = 0;
        let mut correct_at_10 = 0;

        for (i, results) in query_results.iter().enumerate() {
            // Get ground truth for this query
            let gt_neighbors: Vec<i32> = neighbors
                .row(i)
                .iter()
                .take(self.k_neighbours)
                .copied()
                .collect();

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

        let results = format!(
            "\n========== Benchmark Results ==========\n\
             \nPerformance Metrics (query execution only):\n\
             --------------------------------------------\n\
             Total time:    {:.2} seconds\n\
             Queries/sec:   {:.0} QPS\n\
             Avg latency:   {:.2} ms/query\n\
             \nAccuracy Metrics:\n\
             -----------------\n\
             Recall@1:      {:.2}%\n\
             Recall@10:     {:.2}%\n\
             \n========================================",
            elapsed.as_secs_f64(),
            qps,
            avg_latency_ms,
            (correct_at_1 as f64 / n_queries as f64) * 100.0,
            (correct_at_10 as f64 / (n_queries * self.k_neighbours) as f64) * 100.0
        );
        info!("{}", results);

        Ok(())
    }
}
