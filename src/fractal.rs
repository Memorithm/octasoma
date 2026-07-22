//! Fractal (zoomable, multi-resolution) memory.
//!
//! The octree is a fractal: it subdivides space into eight self-similar cells,
//! recursively, down to `min_half_size` — an almost-infinite "zoom". OctaSoma's
//! flat $k$-NN only ever reads the *leaves*. This module exposes the **whole
//! hierarchy**: every depth is a *zoom level*, coarse near the root and finer
//! toward the leaves. You can summarise the region a query falls in at any
//! resolution, and walk the coarse→fine path — navigating memory the way you
//! zoom into a fractal image to reveal more detail.
//!
//! This multi-resolution view is what distinguishes OctaSoma from using an octree
//! as a flat spatial index: the tree is the memory's *map*, not just its index.

use crate::{FractalMemory3D, NONE, NodeId};

/// A summary of one octree region at a chosen zoom depth.
#[derive(Clone, Debug)]
pub struct RegionView {
    /// Depth from the root (0 = the whole memory; larger = deeper zoom).
    pub level: u32,
    /// Centre of this region's cube (in projected 3-D space).
    pub center: [f32; 3],
    /// Half-side of the cube; smaller means a more zoomed-in region.
    pub half_size: f32,
    /// Number of memories contained in this region (the whole subtree).
    pub count: usize,
    /// Mean projected point of the contained memories — the region's "gist".
    pub centroid: [f32; 3],
    /// A few representative payloads from the region (capped).
    pub samples: Vec<Vec<u8>>,
}

impl FractalMemory3D {
    /// Zooms to the region containing `embedding` at depth `level` and summarises
    /// everything beneath it. `level = 0` is the whole memory; deeper levels are
    /// finer. Descent stops early at a leaf or an empty octant. Returns `None`
    /// only for a wrong-dimension or non-finite query.
    pub fn zoom(&self, embedding: &[f32], level: u32, max_samples: usize) -> Option<RegionView> {
        let p = self.project(embedding)?;
        if !p.iter().all(|c| c.is_finite()) {
            return None;
        }
        let (node, depth) = self.descend_along(p, level);
        Some(self.summarize(node, depth, max_samples))
    }

    /// The coarse→fine sequence of region summaries from the root down to
    /// `max_level` along `embedding` — i.e. the act of zooming in, step by step.
    /// Region counts are non-increasing as you descend.
    pub fn zoom_path(
        &self,
        embedding: &[f32],
        max_level: u32,
        max_samples: usize,
    ) -> Vec<RegionView> {
        let p = match self.project(embedding) {
            Some(p) if p.iter().all(|c| c.is_finite()) => p,
            _ => return Vec::new(),
        };
        let mut out = Vec::new();
        let mut node: NodeId = 0;
        let mut depth = 0u32;
        loop {
            out.push(self.summarize(node, depth, max_samples));
            if depth >= max_level {
                break;
            }
            let n = &self.nodes[node as usize];
            if n.is_leaf() {
                break;
            }
            let child = n.children[Self::octant_index(n.center, p)];
            if child == NONE {
                break;
            }
            node = child;
            depth += 1;
        }
        out
    }

    fn descend_along(&self, p: [f32; 3], level: u32) -> (NodeId, u32) {
        let mut node: NodeId = 0;
        let mut depth = 0u32;
        while depth < level {
            let n = &self.nodes[node as usize];
            if n.is_leaf() {
                break;
            }
            let child = n.children[Self::octant_index(n.center, p)];
            if child == NONE {
                break;
            }
            node = child;
            depth += 1;
        }
        (node, depth)
    }

    fn summarize(&self, node_id: NodeId, depth: u32, max_samples: usize) -> RegionView {
        let node = &self.nodes[node_id as usize];
        let mut count = 0usize;
        let mut sum = [0f64; 3];
        let mut samples = Vec::new();
        self.collect_region(node_id, &mut count, &mut sum, &mut samples, max_samples);
        let centroid = if count > 0 {
            let c = count as f64;
            [
                (sum[0] / c) as f32,
                (sum[1] / c) as f32,
                (sum[2] / c) as f32,
            ]
        } else {
            node.center
        };
        RegionView {
            level: depth,
            center: node.center,
            half_size: node.half_size,
            count,
            centroid,
            samples,
        }
    }

    fn collect_region(
        &self,
        node_id: NodeId,
        count: &mut usize,
        sum: &mut [f64; 3],
        samples: &mut Vec<Vec<u8>>,
        max_samples: usize,
    ) {
        let node = &self.nodes[node_id as usize];
        if node.is_leaf() {
            for &item in &self.leaf_buckets[node.bucket_id as usize] {
                let it = &self.items[item as usize];
                *count += 1;
                sum[0] += it.point[0] as f64;
                sum[1] += it.point[1] as f64;
                sum[2] += it.point[2] as f64;
                if samples.len() >= max_samples {
                    continue;
                }
                if let Some(pl) = self.get_payload(item) {
                    samples.push(pl.to_vec());
                }
            }
            return;
        }
        for &child in &node.children {
            if child != NONE {
                self.collect_region(child, count, sum, samples, max_samples);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DeterministicRng, FractalMemory3D};

    fn populated() -> FractalMemory3D {
        let mut m = FractalMemory3D::new(16, 1);
        let mut rng = DeterministicRng::new(7);
        for i in 0..4000 {
            let v: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
            m.insert(&v, Some(format!("item-{i}").as_bytes())).unwrap();
        }
        m
    }

    #[test]
    fn root_zoom_covers_everything() {
        let m = populated();
        let probe: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01).collect();
        let root = m.zoom(&probe, 0, 3).unwrap();
        assert_eq!(root.level, 0);
        assert_eq!(root.count, m.item_count());
    }

    #[test]
    fn zooming_in_narrows_the_region() {
        let m = populated();
        let probe: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let path = m.zoom_path(&probe, 12, 2);
        assert!(path.len() >= 2, "expected to descend at least one level");
        // Counts are non-increasing and half-size strictly shrinks as we zoom in.
        for w in path.windows(2) {
            assert!(
                w[1].count <= w[0].count,
                "count must not grow when zooming in"
            );
            assert!(
                w[1].half_size < w[0].half_size,
                "cube must shrink when zooming in"
            );
        }
        // The deepest region is a strict subset of the whole memory.
        assert!(path.last().unwrap().count <= path[0].count);
        assert_eq!(path[0].count, m.item_count());
    }

    #[test]
    fn zoom_handles_empty_and_bad_input() {
        let empty = FractalMemory3D::new(8, 0);
        let v = [0.1f32; 8];
        assert_eq!(empty.zoom(&v, 0, 1).unwrap().count, 0);
        assert!(empty.zoom_path(&v, 5, 1).len() == 1); // just the (empty) root
        // wrong dimension / non-finite
        assert!(empty.zoom(&[0.0; 3], 0, 1).is_none());
        assert!(empty.zoom(&[f32::NAN; 8], 0, 1).is_none());
    }
}
