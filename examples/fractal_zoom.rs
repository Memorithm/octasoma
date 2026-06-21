//! Fractal memory — zooming from the whole memory down to a single recollection.
//!
//! The octree is self-similar at every scale; each depth is a zoom level. We walk
//! the coarse→fine path a query falls through, printing how the region narrows —
//! the "zoom into a fractal image to reveal finer data" idea, applied to memory.
//!
//! Run: `cargo run --release --example fractal_zoom`

use octasoma::{DeterministicRng, FractalMemory3D};

fn main() {
    let d = 64;
    let themes = 8;
    let mut rng = DeterministicRng::new(0xF00D);

    // A few thematic centres; each memory is "theme T, note N".
    let centers: Vec<Vec<f32>> = (0..themes).map(|_| unit(rand_vec(&mut rng, d))).collect();
    let data: Vec<Vec<f32>> = (0..20_000)
        .map(|i| unit(noisy(&centers[i % themes], 0.3, &mut rng)))
        .collect();

    // Calibrate the 3-D projection with PCA so regions are theme-coherent — the
    // zoom then reveals *topical* structure, not just spatial cells.
    let calib: Vec<f32> = data[..4_000].iter().flatten().copied().collect();
    let mut mem = FractalMemory3D::new_with_pca(d, &calib, 4_000);
    for (i, v) in data.iter().enumerate() {
        mem.insert(
            v,
            Some(format!("theme {} · note {i}", i % themes).as_bytes()),
        )
        .unwrap();
    }
    println!(
        "stored {} memories across {themes} themes in {} octree nodes\n",
        mem.item_count(),
        mem.node_count()
    );

    // Pick a query near one theme and zoom in along it.
    let query = unit(noisy(&centers[3], 0.3, &mut rng));
    println!("zooming in along a query near theme 3:\n");
    println!(
        "{:>5} {:>8} {:>11} {:>34}",
        "level", "count", "half_size", "one sample memory in region"
    );
    println!("{}", "-".repeat(64));
    for view in mem.zoom_path(&query, 16, 1) {
        let sample = view
            .samples
            .first()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_else(|| "—".into());
        println!(
            "{:>5} {:>8} {:>11.4} {:>34}",
            view.level, view.count, view.half_size, sample
        );
    }

    println!(
        "\nEach step is a deeper zoom: the region shrinks and the memory count drops\n\
         from the whole store to the handful nearest the query — coarse theme → exact note.\n\
         The agent can stop at any resolution: a broad 'what do I know near here?' or a\n\
         precise recollection."
    );
}

fn rand_vec(rng: &mut DeterministicRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32()).collect()
}

fn noisy(center: &[f32], spread: f32, rng: &mut DeterministicRng) -> Vec<f32> {
    center
        .iter()
        .map(|&c| c + spread * rng.next_f32())
        .collect()
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}
