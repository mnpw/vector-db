fn main() {
    println!("Hello, world!");
}

type Vector = Vec<f64>;

struct DB {
    inner: Vec<Vector>,
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

        // 2. sort: Euclidean = smallest first, Dot = largest first
        let cmp = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1);
        match self.distance_metric {
            Distance::Euclidean => distances.sort_by(cmp),
            Distance::Dot => distances.sort_by(cmp),
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
