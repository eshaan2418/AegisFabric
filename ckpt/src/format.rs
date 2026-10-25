//! On-disk and wire formats (protocol P1, docs/protocols.md).
//!
//! Segment record:  `MAGIC(4) | len(u32) | chunk_id(32) | blake3(payload)(32) | payload(len) | crc32c(header)(4)`
//! WAL record:      `MAGIC(4) | seq(u64) | len(u32) | manifest(len) | blake3(record)(32)`
//! Manifest (server view; the body is opaque, the header is cleartext):
//!   `MAGIC(4) | name_len(u16) | name | stream_len(u16) | stream | mseq(u64) | n(u32) | ids(n*32) | body_len(u32) | body`
//!
//! All integers are little-endian.

use anyhow::{bail, ensure, Result};

pub const SEG_MAGIC: &[u8; 4] = b"AGC1";
pub const WAL_MAGIC: &[u8; 4] = b"AGW1";
pub const MAN_MAGIC: &[u8; 4] = b"AGM1";

pub const SEG_HDR: usize = 4 + 4 + 32 + 32;
pub const SEG_TRAILER: usize = 4;
pub const WAL_HDR: usize = 4 + 8 + 4;
pub const WAL_TRAILER: usize = 32;
/// Upper bound on a single chunk (defensive parsing of possibly-corrupt length fields).
pub const MAX_CHUNK: usize = 64 << 20;
/// Upper bound on a single manifest.
pub const MAX_MANIFEST: usize = 256 << 20;

pub type ChunkId = [u8; 32];

pub fn hex(id: &[u8]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

/// Encoded size of a segment record carrying `len` payload bytes.
pub fn seg_record_len(len: usize) -> usize {
    SEG_HDR + len + SEG_TRAILER
}

pub fn encode_seg_record(id: &ChunkId, payload: &[u8], out: &mut Vec<u8>) {
    let start = out.len();
    out.extend_from_slice(SEG_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(blake3::hash(payload).as_bytes());
    let crc = crc32c::crc32c(&out[start..start + SEG_HDR]);
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc.to_le_bytes());
}

/// A parsed segment record header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegHeader {
    pub len: u32,
    pub id: ChunkId,
    pub hash: [u8; 32],
}

/// Parse the fixed-size header. Does not check the trailing CRC (see `check_seg_crc`).
pub fn parse_seg_header(h: &[u8]) -> Option<SegHeader> {
    if h.len() < SEG_HDR || &h[0..4] != SEG_MAGIC {
        return None;
    }
    let len = u32::from_le_bytes(h[4..8].try_into().unwrap());
    if len as usize > MAX_CHUNK {
        return None;
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&h[8..40]);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&h[40..72]);
    Some(SegHeader { len, id, hash })
}

pub fn check_seg_crc(header: &[u8], trailer: &[u8]) -> bool {
    trailer.len() == 4
        && crc32c::crc32c(&header[..SEG_HDR]) == u32::from_le_bytes(trailer.try_into().unwrap())
}

/// Fully verify a complete record read from disk; returns the payload slice.
pub fn verify_seg_record<'a>(rec: &'a [u8], want_id: &ChunkId, want_len: u32) -> Result<&'a [u8]> {
    let h = parse_seg_header(rec).ok_or_else(|| anyhow::anyhow!("bad segment record header"))?;
    ensure!(
        h.id == *want_id,
        "segment record holds a different chunk id"
    );
    ensure!(h.len == want_len, "segment record length mismatch");
    let n = h.len as usize;
    ensure!(rec.len() == seg_record_len(n), "short segment record");
    ensure!(
        check_seg_crc(&rec[..SEG_HDR], &rec[SEG_HDR + n..]),
        "segment header crc mismatch"
    );
    let payload = &rec[SEG_HDR..SEG_HDR + n];
    ensure!(
        blake3::hash(payload).as_bytes() == &h.hash,
        "chunk payload blake3 mismatch"
    );
    Ok(payload)
}

pub fn encode_wal_record(seq: u64, manifest: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(WAL_HDR + manifest.len() + WAL_TRAILER);
    out.extend_from_slice(WAL_MAGIC);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
    out.extend_from_slice(manifest);
    let h = blake3::hash(&out);
    out.extend_from_slice(h.as_bytes());
    out
}

/// Try to parse one WAL record at the start of `buf`. Returns (seq, manifest range, total length).
pub fn parse_wal_record(buf: &[u8]) -> Option<(u64, std::ops::Range<usize>, usize)> {
    if buf.len() < WAL_HDR + WAL_TRAILER || &buf[0..4] != WAL_MAGIC {
        return None;
    }
    let seq = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    let len = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    if len > MAX_MANIFEST {
        return None;
    }
    let total = WAL_HDR + len + WAL_TRAILER;
    if buf.len() < total {
        return None;
    }
    let h = blake3::hash(&buf[..WAL_HDR + len]);
    if h.as_bytes() != &buf[WAL_HDR + len..total] {
        return None;
    }
    Some((seq, WAL_HDR..WAL_HDR + len, total))
}

/// Cleartext header of a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestHeader {
    pub name: String,
    pub stream: String,
    pub mseq: u64,
    pub ids: Vec<ChunkId>,
}

pub fn encode_manifest(h: &ManifestHeader, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAN_MAGIC);
    out.extend_from_slice(&(h.name.len() as u16).to_le_bytes());
    out.extend_from_slice(h.name.as_bytes());
    out.extend_from_slice(&(h.stream.len() as u16).to_le_bytes());
    out.extend_from_slice(h.stream.as_bytes());
    out.extend_from_slice(&h.mseq.to_le_bytes());
    out.extend_from_slice(&(h.ids.len() as u32).to_le_bytes());
    for id in &h.ids {
        out.extend_from_slice(id);
    }
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "manifest truncated");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

pub fn parse_manifest(b: &[u8]) -> Result<ManifestHeader> {
    let mut r = Rd { b, p: 0 };
    if r.take(4)? != MAN_MAGIC {
        bail!("bad manifest magic");
    }
    let n = r.u16()? as usize;
    let name = String::from_utf8(r.take(n)?.to_vec())?;
    let n = r.u16()? as usize;
    let stream = String::from_utf8(r.take(n)?.to_vec())?;
    let mseq = r.u64()?;
    let n = r.u32()? as usize;
    ensure!(n <= (b.len() / 32) + 1, "manifest chunk count implausible");
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        let mut id = [0u8; 32];
        id.copy_from_slice(r.take(32)?);
        ids.push(id);
    }
    let bl = r.u32()? as usize;
    r.take(bl)?;
    ensure!(r.p == b.len(), "trailing bytes after manifest body");
    Ok(ManifestHeader {
        name,
        stream,
        mseq,
        ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seg_roundtrip_and_corruption() {
        let id = [7u8; 32];
        let mut v = Vec::new();
        encode_seg_record(&id, b"hello world", &mut v);
        assert_eq!(v.len(), seg_record_len(11));
        assert_eq!(verify_seg_record(&v, &id, 11).unwrap(), b"hello world");
        for i in 0..v.len() {
            let mut w = v.clone();
            w[i] ^= 0x10;
            assert!(
                verify_seg_record(&w, &id, 11).is_err(),
                "flip at byte {i} undetected"
            );
        }
    }

    #[test]
    fn wal_roundtrip_and_torn() {
        let r = encode_wal_record(5, b"manifest-bytes");
        let (seq, range, total) = parse_wal_record(&r).unwrap();
        assert_eq!(
            (seq, &r[range], total),
            (5, &b"manifest-bytes"[..], r.len())
        );
        for cut in 0..r.len() {
            assert!(parse_wal_record(&r[..cut]).is_none());
        }
        for i in 0..r.len() {
            let mut w = r.clone();
            w[i] ^= 1;
            assert!(
                parse_wal_record(&w).is_none(),
                "flip at byte {i} undetected"
            );
        }
    }

    #[test]
    fn manifest_roundtrip() {
        let h = ManifestHeader {
            name: "k".into(),
            stream: "s".into(),
            mseq: 9,
            ids: vec![[1; 32], [2; 32]],
        };
        let m = encode_manifest(&h, b"opaque");
        assert_eq!(parse_manifest(&m).unwrap(), h);
        assert!(parse_manifest(&m[..m.len() - 1]).is_err());
    }
}
