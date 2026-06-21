//! Explainability — the "why" behind a recall, and a visualization export.
//!
//! Because OctaSoma is natively 3-D, every memory has a real position you can
//! show. [`FractalMemory3D::explain`] turns a query into a human-inspectable
//! account: where it landed in space, the nested fractal regions it falls through
//! (coarse→fine), and the nearest memories with their distances and coordinates.
//! [`FractalMemory3D::export_points_json`] dumps the memory for a 3-D viewer — a
//! memory you can literally look at, not a black-box high-D index.

use crate::FractalMemory3D;
use crate::fractal::RegionView;

/// One retrieved memory with its spatial context.
#[derive(Clone, Debug)]
pub struct Neighbor {
    /// The stored payload bytes.
    pub payload: Vec<u8>,
    /// Euclidean distance to the query in the projected 3-D space.
    pub distance: f32,
    /// Where this memory sits in 3-D.
    pub point: [f32; 3],
}

/// A human-inspectable account of a recall.
#[derive(Clone, Debug)]
pub struct Explanation {
    /// The query's projected 3-D location.
    pub query_point: [f32; 3],
    /// The nearest memories, nearest first.
    pub neighbors: Vec<Neighbor>,
    /// The coarse→fine regions the query falls through (the spatial "why").
    pub zoom_path: Vec<RegionView>,
}

impl FractalMemory3D {
    /// Explains a recall: the query's 3-D position, the nested regions it falls
    /// through, and the `k` nearest memories (payload, distance, position).
    /// Returns `None` only for a wrong-dimension or non-finite query.
    pub fn explain(&self, embedding: &[f32], k: usize) -> Option<Explanation> {
        let query_point = self.project(embedding)?;
        if !query_point.iter().all(|c| c.is_finite()) {
            return None;
        }
        let neighbors = self
            .nearest_embedding(embedding, k)
            .into_iter()
            .map(|(id, d2)| Neighbor {
                payload: self.get_payload(id).map(|p| p.to_vec()).unwrap_or_default(),
                distance: d2.max(0.0).sqrt(),
                point: self.items[id as usize].point,
            })
            .collect();
        let zoom_path = self.zoom_path(embedding, 24, 1);
        Some(Explanation {
            query_point,
            neighbors,
            zoom_path,
        })
    }

    /// Exports up to `max_points` memories as a JSON object
    /// `{count, half_size, points:[{x,y,z,payload}]}` for a 3-D scatter viewer.
    /// Dependency-free; payloads are truncated to keep the file small.
    pub fn export_points_json(&self, max_points: usize) -> String {
        let mut out = String::from("{\"count\":");
        out.push_str(&self.item_count().to_string());
        out.push_str(",\"half_size\":");
        out.push_str(&format!("{:.6}", self.world_half_size));
        out.push_str(",\"points\":[");
        let n = self.item_count().min(max_points);
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            let p = self.items[i].point;
            let payload = self.get_payload(i as crate::ItemId).unwrap_or(b"");
            let text = String::from_utf8_lossy(payload);
            let truncated: String = text.chars().take(120).collect();
            out.push_str(&format!(
                "{{\"x\":{:.6},\"y\":{:.6},\"z\":{:.6},\"payload\":{}}}",
                p[0],
                p[1],
                p[2],
                json_string(&truncated)
            ));
        }
        out.push_str("]}");
        out
    }

    /// Like [`FractalMemory3D::export_points_json`], but tags each point with a
    /// `score` from `scores` (aligned by item id) and sets `"scored":true`, so the
    /// viewer can heat-colour by score (e.g. cosine similarity to a query). Missing
    /// scores default to `0`.
    pub fn export_points_json_scored(&self, scores: &[f32], max_points: usize) -> String {
        let mut out = String::from("{\"count\":");
        out.push_str(&self.item_count().to_string());
        out.push_str(",\"half_size\":");
        out.push_str(&format!("{:.6}", self.world_half_size));
        out.push_str(",\"scored\":true,\"points\":[");
        let n = self.item_count().min(max_points);
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            let p = self.items[i].point;
            let payload = self.get_payload(i as crate::ItemId).unwrap_or(b"");
            let text = String::from_utf8_lossy(payload);
            let truncated: String = text.chars().take(120).collect();
            let score = scores.get(i).copied().unwrap_or(0.0);
            out.push_str(&format!(
                "{{\"x\":{:.6},\"y\":{:.6},\"z\":{:.6},\"score\":{:.4},\"payload\":{}}}",
                p[0],
                p[1],
                p[2],
                score,
                json_string(&truncated)
            ));
        }
        out.push_str("]}");
        out
    }
}

/// Escapes a string as a JSON string literal (including surrounding quotes).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use crate::{DeterministicRng, FractalMemory3D};

    fn populated() -> FractalMemory3D {
        let mut m = FractalMemory3D::new(16, 3);
        let mut rng = DeterministicRng::new(11);
        for i in 0..2000 {
            let v: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
            m.insert(&v, Some(format!("memory {i}").as_bytes()))
                .unwrap();
        }
        m
    }

    #[test]
    fn explain_gives_neighbors_and_zoom_path() {
        let m = populated();
        let q: Vec<f32> = (0..16).map(|i| (i as f32).cos()).collect();
        let e = m.explain(&q, 5).unwrap();
        assert!(e.query_point.iter().all(|c| c.is_finite()));
        assert_eq!(e.neighbors.len(), 5);
        // Distances are non-negative and sorted nearest-first.
        for w in e.neighbors.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-6);
            assert!(w[0].distance >= 0.0);
        }
        // The zoom path starts at the whole memory and narrows.
        assert_eq!(e.zoom_path[0].count, m.item_count());
        assert!(e.zoom_path.last().unwrap().count <= e.zoom_path[0].count);
    }

    #[test]
    fn explain_rejects_bad_input() {
        let m = populated();
        assert!(m.explain(&[0.0; 4], 3).is_none()); // wrong dim
        assert!(m.explain(&[f32::NAN; 16], 3).is_none()); // non-finite
    }

    #[test]
    fn export_json_is_well_formed_ish() {
        let m = populated();
        let json = m.export_points_json(50);
        assert!(json.starts_with("{\"count\":2000"));
        assert!(json.contains("\"points\":["));
        assert!(json.contains("\"x\":"));
        assert!(json.contains("\"payload\":\"memory "));
        assert!(json.ends_with("]}"));
        // 50 points → 49 separating commas inside the array region.
        let commas = json.matches("},{").count();
        assert_eq!(commas, 49);
    }
}
