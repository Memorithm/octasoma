from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 anchor, found {count}")
    return text.replace(old, new, 1)


p = Path("src/sketch.rs")
s = p.read_text()
old = '''    /// Core of [`SketchIndex::nearest`], on item ids (insertion order) instead of
    /// payloads — also the exact pipeline [`SketchIndex::certify_shortlist`] measures.
    fn nearest_ids(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dim
            || query.iter().any(|x| !x.is_finite())
            || k == 0
            || self.is_empty()
        {
            return Vec::new();
        }
        let qs = self.sketch_with_path(query);
        let m = shortlist.max(k).min(self.len());

        // 1. Hamming shortlist of size m.
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(&qs, self.sketch_of(i)), i))
            .collect();
        if cand.len() > m {
            cand.select_nth_unstable_by_key(m - 1, |(h, i)| (*h, *i));
            cand.truncate(m);
        }

        // 2. Exact rerank of the shortlist (query prepared once for the store's
        //    precision; each candidate costs a single dot — f32 or integer).
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> =
            cand.iter().map(|&(_, i)| (self.score(i, &q), i)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored.into_iter().map(|(s, i)| (i, s)).collect()
    }
'''
new = '''    /// Computes the query sketch once with this store's persisted compute path.
    /// Shared-projector sharded stores can reuse the result across regions.
    pub(crate) fn query_sketch(&self, query: &[f32]) -> Option<Vec<u64>> {
        if query.len() != self.dim || query.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(self.sketch_with_path(query))
    }

    /// Precise recall using a caller-supplied sketch computed by an equivalent
    /// [`SketchIndex`] (same projector and scalar/SIMD sketch path).
    pub(crate) fn nearest_with_sketch(
        &self,
        query: &[f32],
        query_sketch: &[u64],
        k: usize,
        shortlist: usize,
    ) -> Vec<(&[u8], f32)> {
        self.nearest_ids_with_sketch(query, query_sketch, k, shortlist)
            .into_iter()
            .map(|(i, score)| (self.payload(i), score))
            .collect()
    }

    /// Core of [`SketchIndex::nearest`], on item ids (insertion order) instead of
    /// payloads — also the exact pipeline [`SketchIndex::certify_shortlist`] measures.
    fn nearest_ids(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(usize, f32)> {
        let Some(query_sketch) = self.query_sketch(query) else {
            return Vec::new();
        };
        self.nearest_ids_with_sketch(query, &query_sketch, k, shortlist)
    }

    fn nearest_ids_with_sketch(
        &self,
        query: &[f32],
        query_sketch: &[u64],
        k: usize,
        shortlist: usize,
    ) -> Vec<(usize, f32)> {
        if query.len() != self.dim
            || query.iter().any(|x| !x.is_finite())
            || query_sketch.len() != self.hasher.words()
            || k == 0
            || self.is_empty()
        {
            return Vec::new();
        }
        let m = shortlist.max(k).min(self.len());

        // 1. Hamming shortlist of size m. The expensive query projection has
        // already been computed by the caller and may be shared across shards.
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(query_sketch, self.sketch_of(i)), i))
            .collect();
        if cand.len() > m {
            cand.select_nth_unstable_by_key(m - 1, |(h, i)| (*h, *i));
            cand.truncate(m);
        }

        // 2. Exact rerank of the shortlist (query prepared once for this shard's
        // precision; each candidate costs a single dot — f32 or integer).
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> =
            cand.iter().map(|&(_, i)| (self.score(i, &q), i)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored.into_iter().map(|(score, i)| (i, score)).collect()
    }
'''
s = once(s, old, new, "nearest_ids refactor")

marker = "    #[test]\n    fn normalize_on_insert_makes_cosine_a_dot() {\n"
test = '''    #[test]
    fn precomputed_query_sketch_matches_regular_precision_recall() {
        let mut idx = SketchIndex::new(8, 128, 41);
        for i in 0..12u8 {
            let mut v = vec![0.0f32; 8];
            v[(i as usize) % 8] = 1.0;
            v[((i as usize) + 3) % 8] = 0.2;
            assert!(idx.insert(&v, &[i]));
        }
        let query = [1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0];
        let sketch = idx.query_sketch(&query).unwrap();
        let normal: Vec<(u8, f32)> = idx
            .nearest(&query, 4, 8)
            .into_iter()
            .map(|(p, s)| (p[0], s))
            .collect();
        let reused: Vec<(u8, f32)> = idx
            .nearest_with_sketch(&query, &sketch, 4, 8)
            .into_iter()
            .map(|(p, s)| (p[0], s))
            .collect();
        assert_eq!(normal, reused);
        assert!(idx.nearest_with_sketch(&query, &[], 4, 8).is_empty());
    }

'''
s = once(s, marker, test + marker, "sketch reuse test")
p.write_text(s)

p = Path("src/hybrid.rs")
s = p.read_text()
old = '''    pub fn recall_global(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        let mut hits: Vec<(String, f32)> = Vec::new();
        for shard in self.shards.values() {
            for (p, s) in shard.query(&v, QueryStrategy::PrecisionSketch, k) {
                hits.push((String::from_utf8_lossy(p).into_owned(), s));
            }
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(k);
        Ok(hits)
    }
'''
new = '''    pub fn recall_global(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        let mut hits: Vec<(String, f32)> = Vec::new();
        let Some(first) = self.shards.values().next() else {
            return Ok(hits);
        };

        // `ShardedHybrid` shares one SimHasher across regions. When every shard
        // also uses the same persisted scalar/SIMD sketch path, project the query
        // exactly once and reuse that bit-vector for every Hamming shortlist.
        let sketch_path = first.sketch.simd_sketching();
        let can_reuse = self
            .shards
            .values()
            .all(|shard| shard.sketch.simd_sketching() == sketch_path);
        let shared_sketch = can_reuse.then(|| first.sketch.query_sketch(&v)).flatten();

        for shard in self.shards.values() {
            let shard_hits = if let Some(query_sketch) = shared_sketch.as_deref() {
                let shortlist = shard.default_shortlist.max(k);
                shard
                    .sketch
                    .nearest_with_sketch(&v, query_sketch, k, shortlist)
            } else {
                shard.query(&v, QueryStrategy::PrecisionSketch, k)
            };
            for (payload, score) in shard_hits {
                hits.push((String::from_utf8_lossy(payload).into_owned(), score));
            }
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(k);
        Ok(hits)
    }
'''
s = once(s, old, new, "global recall reuse")

marker = "    #[test]\n    fn sharded_hybrid_shares_one_simhash_projector_across_regions_and_reload() {\n"
test = '''    #[test]
    fn global_recall_matches_per_shard_precision_results_with_shared_query_sketch() {
        let mut mem = ShardedHybrid::new(crate::HashEmbedder::new(32), 128);
        mem.insert("a", "a:alpha", "alpha durable memory").unwrap();
        mem.insert("b", "b:beta", "beta durable memory").unwrap();
        mem.insert("c", "c:gamma", "gamma durable memory").unwrap();

        let global = mem.recall_global("beta durable memory", 3).unwrap();
        assert_eq!(global.len(), 3);
        assert_eq!(global[0].0, "b:beta");

        let mut manual = Vec::new();
        for region in ["a", "b", "c"] {
            manual.extend(mem.recall(region, "beta durable memory", 3).unwrap());
        }
        manual.sort_by(|a, b| b.1.total_cmp(&a.1));
        manual.truncate(3);
        assert_eq!(global, manual);
    }

'''
s = once(s, marker, test + marker, "global recall test")
p.write_text(s)
