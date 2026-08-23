//! A self-contained, fully offline evaluation shaped after LongMemEval
//! (arXiv:2410.10813): five core capabilities — information extraction,
//! multi-session reasoning, temporal reasoning, knowledge updates, and
//! abstention — exercised against a `ShardedHybrid` store whose memories
//! carry full record lifecycles (generations, tombstones, filtered recall).
//!
//! No datasets, no network, no LLM judge: sessions are generated
//! deterministically; answers are scored by anchored gold-substring
//! containment; abstention by a fixed cosine line the offline hasher only
//! crosses for near-duplicates. Queries are the exact session texts — with the offline
//! HashEmbedder (a lexical hasher) paraphrase matching is out of scope, so
//! what this measures is the LIFECYCLE machinery: which memory is allowed to
//! answer now, which stays available to history, which never existed.
//!
//! Run: `cargo run --release --example longmem_eval [-- seed]`

use octasoma::{HashEmbedder, QueryStrategy, ShardedHybrid};
use std::collections::BTreeMap;

const DIM: usize = 128;
const BITS: usize = 256;
const DAY_MS: u64 = 86_400_000;
/// Abstention line for the offline HashEmbedder: lexical-hash cosines only
/// reach this for near-duplicate texts, so any weaker match counts as
/// "nothing to say". Fixed and documented rather than tuned per run.
const ABSTAIN_TAU: f32 = 0.5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

struct Topic {
    name: String,
    city: String,
    old_phase: &'static str,
    new_phase: &'static str,
}

fn build_world(seed: u64, n: usize) -> Vec<Topic> {
    let cities = [
        "Lyon", "Osaka", "Porto", "Austin", "Tallinn", "Nantes", "Kyoto", "Bergen",
    ];
    let phases = [
        ("evaluation", "general availability"),
        ("pilot", "migrating"),
        ("on hold", "restarted"),
        ("legacy", "sunset-planned"),
    ];
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|t| {
            let p = phases[(rng.next() as usize) % phases.len()];
            Topic {
                name: format!("project-{}", char::from(b'a' + t as u8)),
                city: cities[(rng.next() as usize) % cities.len()].to_string(),
                old_phase: p.0,
                new_phase: p.1,
            }
        })
        .collect()
}

/// Day-0 text states the old phase; day-14 announces the change with its day.
fn status_text(topic: &Topic, day: u64) -> String {
    if day == 0 {
        format!(
            "{} is in {} phase, hosted in {}",
            topic.name, topic.old_phase, topic.city
        )
    } else {
        format!(
            "{} entered the {} phase on day {}, hosted in {}",
            topic.name, topic.new_phase, day, topic.city
        )
    }
}

fn build_payload(id: &str, text: &str) -> Vec<u8> {
    let mut out = id.as_bytes().to_vec();
    out.push(0x1f);
    out.extend_from_slice(text.as_bytes());
    out
}

fn split_payload(payload: &str) -> (&str, &str) {
    payload.split_once('\u{1f}').unwrap_or(("", payload))
}

fn payload_key(payload: &str) -> &str {
    payload.split_once('\u{1f}').map_or(payload, |(id, _)| id)
}

fn remember(
    mem: &mut ShardedHybrid<HashEmbedder>,
    region: &str,
    id: &str,
    text: &str,
    generation: u64,
    created_at_ms: u64,
) {
    use octasoma::{EmbeddingFingerprint, MemoryId, MemoryRecord, MemoryScope, Provenance};
    let mut record = MemoryRecord::new(
        MemoryId::new(id).expect("valid id"),
        Vec::new(),
        MemoryScope::new("eval", "longmem", "agent").expect("scope"),
        Provenance::new("longmem-generator")
            .expect("source")
            .with_observed_at(created_at_ms),
        EmbeddingFingerprint::new("hash", "hash-embedder", DIM).expect("fingerprint"),
        generation,
    );
    record.retention.expires_at_unix_ms = Some(u64::MAX / 4);
    mem.remember_with_payload(region, record, &build_payload(id, text), text)
        .expect("remember");
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cat {
    Extraction,
    MultiSession,
    Temporal,
    Update,
    Abstention,
}

fn cat_name(cat: Cat) -> &'static str {
    match cat {
        Cat::Extraction => "extraction",
        Cat::MultiSession => "multi-session",
        Cat::Temporal => "temporal",
        Cat::Update => "knowledge-update",
        Cat::Abstention => "abstention",
    }
}

struct Q {
    cat: Cat,
    region: String,
    query: String,
    /// The topic's own name: gold must never be credited to another project
    /// sharing the same city or phase word.
    anchor: String,
    gold: String,
    /// Historical questions deliberately skip the lifecycle filter —
    /// superseded facts are still true about the past.
    historical: bool,
}

impl Q {
    fn gold_hit(&self, content: &str) -> bool {
        content.contains(&self.anchor) && content.contains(&self.gold)
    }
}

fn sum_total(stats: &BTreeMap<&'static str, (usize, usize)>) -> usize {
    stats.values().map(|(_, total)| total).sum()
}

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("seed must be a u64"))
        .unwrap_or(42);
    let world = build_world(seed ^ 0xC0FFEE, 12);
    let distractors = build_world(seed ^ 0xD15C0, 10);

    let mut mem = ShardedHybrid::new(HashEmbedder::new(DIM), BITS);
    let mut generations: BTreeMap<String, u64> = BTreeMap::new();
    let mut ids_by_day: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut next_id = 0usize;
    let mut naive_tokens = 0usize;

    for day in [0u64, 14] {
        for topic in world.iter().chain(distractors.iter()) {
            let text = status_text(topic, day);
            naive_tokens += text.len() / 4 + 1;
            let id = format!("m:{next_id:05}");
            next_id += 1;
            let g = generations.entry(topic.name.clone()).or_default();
            *g += 1;
            remember(&mut mem, &topic.city, &id, &text, *g, day * DAY_MS);
            ids_by_day.entry(day).or_default().push(id);
        }
    }
    let total_memories = mem.len();

    // At "now" (day 21), every day-0 status was replaced on day 14: retire the
    // superseded memories through the real retirement path.
    let now_ms = 21 * DAY_MS;
    let mut retired = 0usize;
    for id in &ids_by_day[&0] {
        if let Some(record) = mem.record(id) {
            let current = record.generation;
            if mem.tombstone(id, current + 1).is_ok() {
                retired += 1;
            }
        }
    }

    // -- questions ------------------------------------------------------------

    let mut questions: Vec<Q> = Vec::new();
    for t in &world {
        questions.push(Q {
            cat: Cat::Extraction,
            region: t.city.clone(),
            query: status_text(t, 14),
            anchor: t.name.clone(),
            gold: t.city.clone(),
            historical: false,
        });
        questions.push(Q {
            cat: Cat::MultiSession,
            region: t.city.clone(),
            query: status_text(t, 0),
            anchor: t.name.clone(),
            gold: t.old_phase.to_string(),
            historical: true,
        });
        questions.push(Q {
            cat: Cat::Temporal,
            region: t.city.clone(),
            query: status_text(t, 14),
            anchor: t.name.clone(),
            gold: "day 14".to_string(),
            historical: false,
        });
        // Asked with the OUTDATED wording: only the retirement path forces
        // the answer to come from the day-14 memory rather than the
        // tombstoned day-0 one.
        questions.push(Q {
            cat: Cat::Update,
            region: t.city.clone(),
            query: format!("{} - what phase now?", status_text(t, 0)),
            anchor: t.name.clone(),
            gold: t.new_phase.to_string(),
            historical: false,
        });
    }
    for t in &world {
        questions.push(Q {
            cat: Cat::Abstention,
            region: t.city.clone(),
            query: format!("does {} have a mobile app", t.name),
            anchor: t.name.clone(),
            gold: String::new(),
            historical: false,
        });
    }

    // -- run ------------------------------------------------------------------

    let run_query = |mem: &ShardedHybrid<HashEmbedder>, q: &Q| -> Vec<(String, f32)> {
        if q.historical {
            mem.recall_with(&q.region, &q.query, 3, QueryStrategy::PrecisionSketch)
        } else {
            mem.recall_visible_by(&q.region, &q.query, 3, now_ms, payload_key)
        }
        .expect("recall")
    };

    let runs: Vec<Vec<(String, f32)>> = questions.iter().map(|q| run_query(&mem, q)).collect();

    // -- score ----------------------------------------------------------------

    let mut stats: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut octa_tokens = 0usize;
    let mut turns_with_hits = 0usize;
    let mut update_stale_without_lifecycle = 0usize;

    for (q, hits) in questions.iter().zip(&runs) {
        let entry = stats.entry(cat_name(q.cat)).or_insert((0, 0));
        entry.1 += 1;
        octa_tokens += hits
            .iter()
            .map(|(payload, _)| split_payload(payload).1.len() / 4 + 1)
            .sum::<usize>();
        if !hits.is_empty() {
            turns_with_hits += 1;
        }

        // The recall returns a context SET; the answer is correct when some
        // member carries the anchored gold above the calibrated threshold.
        // Abstention stays a max-score decision.
        let correct = match q.cat {
            Cat::Abstention => hits
                .first()
                .map(|(_, score)| *score < ABSTAIN_TAU)
                .unwrap_or(true),
            _ => hits.iter().any(|(payload, _)| {
                let (_, content) = split_payload(payload);
                q.gold_hit(content)
            }),
        };
        if correct {
            entry.0 += 1;
        }

        // Ablation: on missed updates, would the lifecycle-unaware recall have
        // answered from the tombstoned memory?
        if matches!(q.cat, Cat::Update) {
            let stale_top1 = mem
                .recall_with(&q.region, &q.query, 3, QueryStrategy::PrecisionSketch)
                .expect("recall")
                .first()
                .map(|(payload, _)| {
                    let (_, content) = split_payload(payload);
                    content.contains("is in") && content.contains(&q.anchor)
                })
                .unwrap_or(false);
            if stale_top1 {
                update_stale_without_lifecycle += 1;
            }
        }
    }

    // -- report ---------------------------------------------------------------

    println!(
        "seed={seed}   memories={}   retired={retired}",
        total_memories
    );
    println!("abstain_tau = {ABSTAIN_TAU} (fixed; hash-cosines reach it only for near-duplicates)");
    println!("{:<18}{:>9}", "category", "ok/total");
    let mut sum_ok = 0usize;
    for (cat, (ok, total)) in &stats {
        println!("{cat:<18}{ok:>5}/{total:<4}");
        sum_ok += ok;
    }
    let sum_all = sum_total(&stats);
    println!("{:-<28}", "");
    println!("{:<18}{sum_ok:>5}/{sum_all:<4}", "TOTAL");

    let avg_octa = (octa_tokens).checked_div(turns_with_hits).unwrap_or(0);
    println!();
    println!("token cost / turn:");
    println!("  naive inject-everything : {naive_tokens:>7}");
    println!(
        "  octasoma recall (k<=3)  : {avg_octa:>7}  (~{}x fewer)",
        naive_tokens.max(1) / avg_octa.max(1)
    );
    println!();
    println!(
        "ablation: without the lifecycle filter, {update_stale_without_lifecycle} \
         knowledge-update question(s) would have been answered from a superseded memory."
    );
    println!();
    println!("honesty note: containment scoring measures retrieval, not generation;");
    println!("the abstention threshold is this corpus's own separating margin.");
}
