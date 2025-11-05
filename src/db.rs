use std::fmt::Display;

use fastembed::{TextEmbedding, TextInitOptions};
use rand::seq::IndexedRandom;

type Vector = Vec<f64>;

// Inverted File Index
pub struct Index {
    inner: Vec<(Vector, Vec<Vector>)>,
    cluster_count: usize,
}

impl Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Index{{clusters={}}}", self.cluster_count)
    }
}

impl Index {
    fn new(k: usize) -> Self {
        Self {
            inner: Vec::with_capacity(k),
            cluster_count: k,
        }
    }

    fn update(&mut self, vectors: &[Vector], distance_metric: &Distance) {
        if vectors.is_empty() {
            return;
        }

        let clusters = k_means(vectors, self.cluster_count, 100, distance_metric);
        self.inner = clusters;
    }

    fn get_clusters(&self) -> &Vec<(Vector, Vec<Vector>)> {
        &self.inner
    }
}

fn k_means(
    vectors: &[Vector],
    k: usize,
    iterations: usize,
    distance_metric: &Distance,
) -> Vec<(Vector, Vec<Vector>)> {
    assert_ne!(k, 0);

    // Initialize K centroids randomly
    let mut centroids: Vec<Vector> = vectors
        .choose_multiple(&mut rand::rng(), k)
        .cloned()
        .collect();

    let mut vector_to_cluster_map: Vec<usize> = vec![0; vectors.len()];

    let mut centroids_changed = true;
    let mut iter = 0;

    while centroids_changed && iter < iterations {
        println!("Building index, iter: {iter}");

        centroids_changed = false;
        iter += 1;

        // Assign vectors to closest centroid
        for (vector_id, vector) in vectors.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            // Compute distance with every centroid
            for (cluster_id, centroid) in centroids.iter().enumerate() {
                assert!(matches!(distance_metric, Distance::Euclidean));
                let dist = distance_metric.compute(vector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_id;
                }
            }

            // Update vector's centroid
            if vector_to_cluster_map[vector_id] != best_cluster {
                // Found a different centroid for ith vector
                vector_to_cluster_map[vector_id] = best_cluster;
                centroids_changed = true;
            }
        }

        // Update centroids using the vector-cluster assignment
        for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
            // Find all vectors belonging to cluster_id

            let mut cluster_vectors_idx = Vec::new();
            for (idx, id) in vector_to_cluster_map.iter().enumerate() {
                if *id == cluster_id {
                    cluster_vectors_idx.push(idx);
                }
            }

            let cluster_vectors: Vec<&Vector> = cluster_vectors_idx
                .into_iter()
                .map(|idx| &vectors[idx])
                .collect();

            // compute centroid
            let mut new_centroid = vec![0.0; vectors[0].len()];
            let cluster_size = cluster_vectors.len();
            if cluster_size > 0 {
                new_centroid = cluster_vectors
                    .iter()
                    // sum every coordinate of cluster vectors
                    .fold(new_centroid, |c, v| {
                        c.iter()
                            .zip(v.iter())
                            .map(|(left, right)| left + right)
                            .collect()
                    })
                    .iter()
                    // divide every coordinate by cluster size
                    .map(|v_i| *v_i / cluster_size as f64)
                    .collect();

                *centroid = new_centroid;
            }
        }
    }

    let mut clusters: Vec<(Vector, Vec<Vector>)> = Vec::new();
    for (cluster_id, centroid) in centroids.into_iter().enumerate() {
        // Find all vectors belonging to cluster_id
        let mut cluster_vectors_idx = Vec::new();
        for (idx, id) in vector_to_cluster_map.iter().enumerate() {
            if *id == cluster_id {
                cluster_vectors_idx.push(idx);
            }
        }

        let cluster_vectors: Vec<Vector> = cluster_vectors_idx
            .into_iter()
            .map(|idx| vectors[idx].to_owned())
            .collect();

        clusters.push((centroid, cluster_vectors))
    }

    clusters
}

#[derive(Clone)]
pub enum Distance {
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

#[derive(Debug)]
pub enum EmbedderOpts {
    Text(TextEmbedderOpts),
}

#[derive(Default, Debug)]
pub struct TextEmbedderOpts {
    inner: TextInitOptions,
}

pub enum Embedder {
    Text(TextEmbedding),
}

impl Embedder {
    fn embed(&mut self, data: String) -> Vector {
        let embeddings = match self {
            Embedder::Text(text_embedding) => text_embedding.embed(vec![data], None).unwrap(),
        };

        // TODO: optimize this
        embeddings
            .first()
            .unwrap()
            .to_owned()
            .into_iter()
            .map(Into::into)
            .collect()
    }
}

pub struct DB {
    pub inner: Vec<Vector>,
    pub index: Index,
    dim: usize,
    distance_metric: Distance,
}

impl DB {
    pub fn new(dimension: usize, distance_metric: Distance) -> Self {
        Self {
            inner: vec![],
            index: Index::new(32),
            dim: dimension,
            distance_metric,
        }
    }

    pub fn new_with_embedder(
        embedder_opts: EmbedderOpts,
        distance_metric: Distance,
    ) -> anyhow::Result<(Self, Embedder)> {
        let opts = match embedder_opts {
            EmbedderOpts::Text(text_embedder_opts) => text_embedder_opts.inner,
        };
        let model_info = TextEmbedding::get_model_info(&opts.model_name)?;
        let dim = model_info.dim;

        let model = TextEmbedding::try_new(Default::default())?;
        let embedder = Embedder::Text(model);

        Ok((
            Self {
                inner: vec![],
                index: Index::new(32),
                dim,
                distance_metric,
            },
            embedder,
        ))
    }

    pub fn build_index(&mut self) {
        if self.inner.len() < (self.index.cluster_count * 256) {
            println!(
                "Not enough vectors ({}), skipping index building",
                self.inner.len()
            );

            return;
        }

        self.index.update(&self.inner, &self.distance_metric);
    }

    pub fn insert(&mut self, vector: Vector) -> Vector {
        assert_eq!(vector.len(), self.dim);
        self.inner.push(vector.clone());
        vector
    }

    pub fn search(&self, vector: &Vector, count: usize) -> Vec<&Vector> {
        assert_eq!(vector.len(), self.dim);

        let clusters = self.index.get_clusters();
        if clusters.is_empty() {
            // Fallback to linear search if no index
            println!("falling back to linear search");
            return self.linear_search(vector, count);
        }

        // Find top clusters by centroid distance
        let mut cluster_distances: Vec<(usize, f64)> = clusters
            .iter()
            .enumerate()
            .map(|(c_idx, (centroid, _))| (c_idx, self.distance_metric.compute(centroid, vector)))
            .collect();

        match self.distance_metric {
            Distance::Euclidean => cluster_distances.sort_by(|a, b| a.1.total_cmp(&b.1)),
            Distance::Dot => cluster_distances.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        // Search within top 10 clusters
        let mut candidates = Vec::new();
        for &(c_idx, _dist) in cluster_distances.iter().take(3) {
            let (_, vectors) = &clusters[c_idx];
            for v in vectors {
                let dist = self.distance_metric.compute(vector, v);
                candidates.push((v, dist));
            }
        }

        // Sort candidates
        match self.distance_metric {
            Distance::Euclidean => candidates.sort_by(|a, b| a.1.total_cmp(&b.1)),
            Distance::Dot => candidates.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        let mut top = Vec::with_capacity(count);
        for (v, _) in candidates.iter().take(count) {
            top.push(*v);
        }

        top
    }

    fn linear_search(&self, vector: &Vector, count: usize) -> Vec<&Vector> {
        let mut distances: Vec<(usize, f64)> = self
            .inner
            .iter()
            .enumerate()
            .map(|(id, v)| (id, self.distance_metric.compute(v, vector)))
            .collect();

        match self.distance_metric {
            Distance::Euclidean => distances.sort_by(|a, b| a.1.total_cmp(&b.1)),
            Distance::Dot => distances.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        let mut top = Vec::with_capacity(count.min(distances.len()));
        for (id, _) in distances.iter().take(count) {
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

#[test]
fn test_embedding() {
    let (mut db, mut embedder) =
        DB::new_with_embedder(EmbedderOpts::Text(Default::default()), Distance::Euclidean).unwrap();
    let mut embed = |input: &'static str| embedder.embed(input.to_string());

    let mut inputs = Vec::new();
    inputs.push(("apple", db.insert(embed("apple"))));
    inputs.push(("banana", db.insert(embed("banana"))));
    inputs.push(("cat", db.insert(embed("cat"))));
    inputs.push(("orange", db.insert(embed("orange"))));

    let outputs = db.search(&embed("dog"), 1);
    let output = outputs.first().unwrap();

    let closest = inputs
        .iter()
        .find_map(|(text, vector)| vector.eq(*output).then_some(*text));

    assert_eq!(closest, Some("cat"));
}
