//! Optional wgpu batch scoring — proposal A5 of `docs/scirust-improvements.md`.
//!
//! The GEMM core is **vendored from SciRust**
//! (`scirust-gpu/src/wgpu_backend.rs`, same org and dual license): depending on
//! scirust-gpu's `wgpu` cargo feature would drag `scirust-core` into the tree,
//! so the ~200 self-contained lines (WGSL shader, device/pipeline setup,
//! dispatch, read-back) live here instead, behind the `gpu` feature
//! (wgpu + pollster + bytemuck — none in the default build).
//!
//! What it is for: the workloads that genuinely score **everything** — viewer
//! heat-maps, cross-shard global recall, benchmark sweeps — where per-query
//! brute force is the documented O(N) ceiling. [`crate::SketchIndex::scores_batch_gpu`]
//! computes all `queries × items` cosines as **one** GEMM (`Q · Eᵀ`, and thanks
//! to the shader's transpose flag, straight off the row-major embedding storage
//! — no transpose copy).
//!
//! The contract, stated plainly:
//!
//! - **Never the default path.** GPU accumulation order is not bit-identical to
//!   the scalar CPU path; results are tolerance-validated (1e-4 relative in the
//!   tests), not bit-exact. Determinism-sensitive callers stay on
//!   [`SketchIndex::scores`](crate::SketchIndex::scores).
//! - **Honest failure.** No Vulkan/Metal/DX12 adapter → [`GpuScorer::new`]
//!   returns `Unsupported` — never fabricated data. CI exercises the real path
//!   on a software Vulkan adapter (Mesa lavapipe), so the claim is tested
//!   without physical GPU hardware.
//! - **F32 tier only.** The quantized tiers trade exactness for memory; adding
//!   a second approximation on top would compound silently.

use std::borrow::Cow;
use std::io;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

/// Row-major GEMM `C(m×n) = op(A)(m×k) · op(B)(k×n)`; `ta`/`tb` flag whether the
/// *stored* buffer is the transpose of `op`. One invocation per output cell.
/// (Vendored verbatim from scirust-gpu, minus the unused `alpha`/`beta` terms.)
const GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var acc: f32 = 0.0;
    for (var q: u32 = 0u; q < p.k; q = q + 1u) {
        var av: f32;
        var bv: f32;
        if (p.ta == 1u) { av = a[q * p.m + i]; } else { av = a[i * p.k + q]; }
        if (p.tb == 1u) { bv = b[j * p.k + q]; } else { bv = b[q * p.n + j]; }
        acc = acc + av * bv;
    }
    c[i * p.n + j] = acc;
}
"#;

fn unavailable(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, what.to_string())
}

/// A wgpu device + the compiled GEMM pipeline, created once and reused across
/// calls (adapter acquisition and shader compilation are the expensive part).
pub struct GpuScorer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
}

impl GpuScorer {
    /// Acquires an adapter/device and compiles the GEMM pipeline. Returns an
    /// [`io::ErrorKind::Unsupported`] error when no adapter exists (e.g. no
    /// Vulkan driver) — never a silent fake.
    pub fn new() -> io::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or_else(|| unavailable("no wgpu adapter (install a Vulkan/Metal/DX12 driver)"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("octasoma-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| unavailable(&format!("wgpu device request failed: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("octasoma-gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("octasoma-gemm"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_name,
        })
    }

    /// The adapter this scorer runs on (e.g. `"llvmpipe (LLVM …)"` on Mesa's
    /// software Vulkan — what CI uses).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// One row-major GEMM `C(m×n) = A(m×k) · op(B)`, with `tb` flagging that the
    /// stored `b` is `op(B)ᵀ` (i.e. `n×k` row-major — exactly octasoma's
    /// embedding layout). Uploads, dispatches, reads back.
    pub(crate) fn gemm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        tb: bool,
    ) -> io::Result<Vec<f32>> {
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if a.len() != m * k || b.len() != k * n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "gemm shape mismatch: |A| = {} vs m*k = {}, |B| = {} vs k*n = {}",
                    a.len(),
                    m * k,
                    b.len(),
                    k * n
                ),
            ));
        }
        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params: [u32; 8] = [m as u32, k as u32, n as u32, 0, tb as u32, 0, 0, 0];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| unavailable("wgpu read-back channel closed"))?
            .map_err(|e| unavailable(&format!("wgpu buffer map failed: {e:?}")))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}
