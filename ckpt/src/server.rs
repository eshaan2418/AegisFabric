//! TCP server (one thread per connection; group commit happens inside the store).
//!
//! Frame: request `op(u8) | len(u32) | body`, response `status(u8) | len(u32) | body`.
//! Status: 0 ok, 1 not found, 2 error (body = UTF-8 message), 3 missing chunks (body = JSON hex list).

use crate::format::ChunkId;
use crate::store::{CommitError, Store};
use anyhow::{bail, ensure, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

pub const OP_PUT_BATCH: u8 = 1;
pub const OP_HAS: u8 = 2;
pub const OP_COMMIT: u8 = 3;
pub const OP_GET_MANIFEST: u8 = 4;
pub const OP_GET_CHUNKS: u8 = 5;
pub const OP_LATEST: u8 = 6;
pub const OP_GC: u8 = 7;
pub const OP_STATS: u8 = 8;
pub const OP_VERIFY: u8 = 9;
pub const OP_GET_MANIFEST_SEQ: u8 = 10;
pub const OP_LIST: u8 = 11;
pub const OP_RECOVERY: u8 = 12;

const MAX_FRAME: usize = 1 << 30;

enum Reply {
    Ok(Vec<u8>),
    NotFound,
    Missing(Vec<ChunkId>),
}

fn json(v: impl serde::Serialize) -> Result<Reply> {
    Ok(Reply::Ok(serde_json::to_vec(&v)?))
}

fn ids(body: &[u8]) -> Result<Vec<ChunkId>> {
    ensure!(
        body.len().is_multiple_of(32),
        "id list not a multiple of 32 bytes"
    );
    Ok(body.as_chunks::<32>().0.to_vec())
}

fn handle(store: &Store, op: u8, body: Vec<u8>) -> Result<Reply> {
    match op {
        OP_PUT_BATCH => {
            ensure!(body.len() >= 4, "short PUT_BATCH");
            let n = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
            let mut p = 4usize;
            let mut chunks = Vec::with_capacity(n);
            for _ in 0..n {
                ensure!(p + 36 <= body.len(), "truncated PUT_BATCH");
                let id: ChunkId = body[p..p + 32].try_into().unwrap();
                let len = u32::from_le_bytes(body[p + 32..p + 36].try_into().unwrap()) as usize;
                p += 36;
                ensure!(p + len <= body.len(), "truncated PUT_BATCH payload");
                chunks.push((id, body[p..p + len].to_vec()));
                p += len;
            }
            json(store.put_batch(chunks)?)
        }
        OP_HAS => Ok(Reply::Ok(
            store.has(&ids(&body)?)?.into_iter().map(u8::from).collect(),
        )),
        OP_COMMIT => match store.commit(body) {
            Ok(r) => json(r),
            Err(CommitError::Missing(m)) => Ok(Reply::Missing(m)),
            Err(CommitError::Other(e)) => Err(e),
        },
        OP_GET_MANIFEST => Ok(store
            .get_manifest(std::str::from_utf8(&body)?)?
            .map_or(Reply::NotFound, |b| Reply::Ok(b.to_vec()))),
        OP_LATEST => Ok(store
            .latest(std::str::from_utf8(&body)?)?
            .map_or(Reply::NotFound, |b| Reply::Ok(b.to_vec()))),
        OP_GET_MANIFEST_SEQ => {
            ensure!(body.len() >= 10, "short GET_MANIFEST_SEQ");
            let n = u16::from_le_bytes(body[0..2].try_into().unwrap()) as usize;
            ensure!(body.len() == 2 + n + 8, "bad GET_MANIFEST_SEQ");
            let stream = std::str::from_utf8(&body[2..2 + n])?;
            let mseq = u64::from_le_bytes(body[2 + n..].try_into().unwrap());
            Ok(store
                .get_manifest_seq(stream, mseq)?
                .map_or(Reply::NotFound, |b| Reply::Ok(b.to_vec())))
        }
        OP_GET_CHUNKS => {
            let data = store.get_chunks(&ids(&body)?)?;
            let mut out = Vec::with_capacity(data.iter().map(|d| d.len() + 4).sum());
            for d in data {
                out.extend_from_slice(&(d.len() as u32).to_le_bytes());
                out.extend_from_slice(&d);
            }
            Ok(Reply::Ok(out))
        }
        OP_GC => {
            let v: serde_json::Value = if body.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_slice(&body)?
            };
            let keep = v.get("keep").and_then(|x| x.as_u64()).unwrap_or(2) as usize;
            let min_dead = v.get("min_dead").and_then(|x| x.as_f64()).unwrap_or(0.3);
            json(store.gc(keep.max(1), min_dead)?)
        }
        OP_STATS => json(store.stats_json()),
        OP_VERIFY => json(store.verify()?),
        OP_LIST => json(store.list()?),
        OP_RECOVERY => json(&store.recovery),
        _ => bail!("unknown op {op}"),
    }
}

fn conn(store: Arc<Store>, s: TcpStream) -> Result<()> {
    s.set_nodelay(true)?;
    let mut r = BufReader::with_capacity(1 << 20, s.try_clone()?);
    let mut w = BufWriter::with_capacity(1 << 20, s);
    loop {
        let mut hdr = [0u8; 5];
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(hdr[1..5].try_into().unwrap()) as usize;
        ensure!(len <= MAX_FRAME, "frame too large");
        let mut body = vec![0u8; len];
        r.read_exact(&mut body)?;
        let (status, out) = match handle(&store, hdr[0], body) {
            Ok(Reply::Ok(b)) => (0u8, b),
            Ok(Reply::NotFound) => (1, Vec::new()),
            Ok(Reply::Missing(m)) => (
                3,
                serde_json::to_vec(&m.iter().map(|i| crate::format::hex(i)).collect::<Vec<_>>())?,
            ),
            Err(e) => (2, format!("{e:#}").into_bytes()),
        };
        w.write_all(&[status])?;
        w.write_all(&(out.len() as u32).to_le_bytes())?;
        w.write_all(&out)?;
        w.flush()?;
    }
}

pub fn serve(store: Arc<Store>, addr: &str, ready_file: Option<&str>) -> Result<()> {
    let l = TcpListener::bind(addr)?;
    let local = l.local_addr()?;
    eprintln!("aegis-ckpt listening on {local} (io={})", store.io_name());
    if let Some(p) = ready_file {
        let tmp = format!("{p}.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec(
                &serde_json::json!({"addr": local.to_string(), "recovery": &store.recovery}),
            )?,
        )?;
        std::fs::rename(&tmp, p)?;
    }
    for s in l.incoming() {
        let s = s?;
        let st = store.clone();
        std::thread::spawn(move || {
            if let Err(e) = conn(st, s) {
                eprintln!("connection error: {e:#}");
            }
        });
    }
    Ok(())
}
