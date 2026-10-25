//! aegis-ckpt CLI.
//!
//!   aegis-ckpt serve --root DIR [--addr 0.0.0.0:7401] [--io auto|uring|pool] [--threads 4] [--seg-mb 64] [--ready-file F]
//!   aegis-ckpt recover --root DIR [--io ...]        open (= crash recovery) and print the recovery report
//!   aegis-ckpt verify --root DIR                    recovery + scrub of every retained manifest
//!   aegis-ckpt chunkscan --mode fixed|cdc --size BYTES FILE...   dedup of successive files

use aegis_ckpt::{chunker, server, store};
use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::sync::Arc;

struct Args {
    v: Vec<String>,
}

impl Args {
    fn opt(&self, k: &str) -> Option<String> {
        self.v
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.v.get(i + 1).cloned())
    }
    fn get(&self, k: &str, d: &str) -> String {
        self.opt(k).unwrap_or_else(|| d.to_string())
    }
    fn positional(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 1;
        while i < self.v.len() {
            if self.v[i].starts_with("--") {
                i += 2;
            } else {
                out.push(self.v[i].clone());
                i += 1;
            }
        }
        out
    }
}

fn config(a: &Args) -> Result<store::Config> {
    let mut c = store::Config::new(a.opt("--root").context("--root is required")?);
    c.io = a.get("--io", "auto");
    c.threads = a.get("--threads", "4").parse()?;
    c.seg_max = a.get("--seg-mb", "64").parse::<u64>()? << 20;
    Ok(c)
}

fn chunkscan(a: &Args) -> Result<()> {
    let mode = a.get("--mode", "fixed");
    let size: usize = a.get("--size", "65536").parse()?;
    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut rows = Vec::new();
    let (mut tot_logical, mut tot_unique) = (0u64, 0u64);
    for f in a.positional() {
        let d = std::fs::read(&f).with_context(|| f.clone())?;
        let ends = match mode.as_str() {
            "fixed" => chunker::fixed(d.len(), size),
            "cdc" => chunker::cdc(&d, size),
            m => bail!("unknown mode {m}"),
        };
        let (mut start, mut new_bytes, mut new_chunks) = (0usize, 0u64, 0u64);
        for &e in &ends {
            let id = *blake3::hash(&d[start..e]).as_bytes();
            if seen.insert(id) {
                new_bytes += (e - start) as u64;
                new_chunks += 1;
            }
            start = e;
        }
        tot_logical += d.len() as u64;
        tot_unique += new_bytes;
        rows.push(
            serde_json::json!({"file": f, "bytes": d.len(), "chunks": ends.len(),
            "new_chunks": new_chunks, "new_bytes": new_bytes}),
        );
    }
    let out = serde_json::json!({"mode": mode, "size": size, "files": rows, "logical_bytes": tot_logical,
        "unique_bytes": tot_unique,
        "dedup_ratio": if tot_unique > 0 { tot_logical as f64 / tot_unique as f64 } else { 0.0 }});
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

fn main() -> Result<()> {
    let a = Args {
        v: std::env::args().skip(1).collect(),
    };
    match a.v.first().map(String::as_str) {
        Some("serve") => {
            let st = Arc::new(store::Store::open(config(&a)?)?);
            server::serve(
                st,
                &a.get("--addr", "0.0.0.0:7401"),
                a.opt("--ready-file").as_deref(),
            )
        }
        Some("recover") => {
            let st = store::Store::open(config(&a)?)?;
            println!(
                "{}",
                serde_json::to_string(
                    &serde_json::json!({"io": st.io_name(), "recovery": &st.recovery})
                )?
            );
            Ok(())
        }
        Some("verify") => {
            let st = store::Store::open(config(&a)?)?;
            let t = std::time::Instant::now();
            let v = st.verify()?;
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({"recovery": &st.recovery, "verify": v,
                "verify_ms": t.elapsed().as_secs_f64() * 1e3}))?
            );
            Ok(())
        }
        Some("chunkscan") => chunkscan(&a),
        _ => bail!("usage: aegis-ckpt serve|recover|verify|chunkscan ... (see source header)"),
    }
}
