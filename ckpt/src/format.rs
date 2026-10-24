//! On-disk record formats (docs/protocols.md P1).
//!
//! Chunk record:  MAGIC(4) | len(u32 LE) | chunk_id(32) | blake3(payload)(32) | payload | crc32c(header)(4)
//! WAL record:    MAGIC(4) | seq(u64 LE) | len(u32 LE) | manifest(len) | blake3(all preceding bytes)(32)

pub const CHUNK_MAGIC: [u8; 4] = *b"AGC1";
pub const WAL_MAGIC: [u8; 4] = *b"AGW1";
pub const CHUNK_HDR: usize = 4 + 4 + 32 + 32;
pub const CHUNK_OVERHEAD: usize = CHUNK_HDR + 4;
pub const WAL_HDR: usize = 4 + 8 + 4;
pub const WAL_OVERHEAD: usize = WAL_HDR + 32;
pub const MAX_CHUNK: usize = 64 << 20;
pub const MAX_MANIFEST: usize = 256 << 20;

pub type Id = [u8; 32];

pub fn encode_chunk(id: &Id, payload: &[u8], out: &mut Vec<u8>) {
    let start = out.len();
    out.extend_from_slice(&CHUNK_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(blake3::hash(payload).as_bytes());
    let crc = crc32c::crc32c(&out[start..start + CHUNK_HDR]);
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc.to_le_bytes());
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChunkCheck {
    Ok,
    /// Not enough bytes for a complete record (torn tail).
    Short,
    /// Complete-looking but invalid (bad magic / crc / hash / id).
    Bad(&'static str),
}

/// Parse a chunk record starting at `buf[0]`. Returns (check, id, payload range, total length).
pub fn parse_chunk(buf: &[u8]) -> (ChunkCheck, Id, std::ops::Range<usize>, usize) {
    let mut id = [0u8; 32];
    if buf.len() < CHUNK_HDR {
        return (ChunkCheck::Short, id, 0..0, 0);
    }
    if buf[0..4] != CHUNK_MAGIC {
        return (ChunkCheck::Bad("magic"), id, 0..0, 0);
    }
    let len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    if len > MAX_CHUNK {
        return (ChunkCheck::Bad("len"), id, 0..0, 0);
    }
    let total = CHUNK_OVERHEAD + len;
    if buf.len() < total {
        return (ChunkCheck::Short, id, 0..0, total);
    }
    id.copy_from_slice(&buf[8..40]);
    let crc = u32::from_le_bytes(buf[CHUNK_HDR + len..total].try_into().unwrap());
    if crc32c::crc32c(&buf[0..CHUNK_HDR]) != crc {
        return (ChunkCheck::Bad("crc"), id, 0..0, total);
    }
    let payload = CHUNK_HDR..CHUNK_HDR + len;
    if blake3::hash(&buf[payload.clone()]).as_bytes() != &buf[40..72] {
        return (ChunkCheck::Bad("payload-hash"), id, 0..0, total);
    }
    (ChunkCheck::Ok, id, payload, total)
}

pub fn encode_wal(seq: u64, manifest: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(WAL_OVERHEAD + manifest.len());
    out.extend_from_slice(&WAL_MAGIC);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
    out.extend_from_slice(manifest);
    let h = blake3::hash(&out);
    out.extend_from_slice(h.as_bytes());
    out
}

/// Parse a WAL record at `buf[0]`: (check, seq, manifest range, total length).
pub fn parse_wal(buf: &[u8]) -> (ChunkCheck, u64, std::ops::Range<usize>, usize) {
    if buf.len() < WAL_HDR {
        return (ChunkCheck::Short, 0, 0..0, 0);
    }
    if buf[0..4] != WAL_MAGIC {
        return (ChunkCheck::Bad("magic"), 0, 0..0, 0);
    }
    let seq = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    let len = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    if len > MAX_MANIFEST {
        return (ChunkCheck::Bad("len"), seq, 0..0, 0);
    }
    let total = WAL_OVERHEAD + len;
    if buf.len() < total {
        return (ChunkCheck::Short, seq, 0..0, total);
    }
    let body = &buf[0..WAL_HDR + len];
    if blake3::hash(body).as_bytes() != &buf[WAL_HDR + len..total] {
        return (ChunkCheck::Bad("checksum"), seq, 0..0, total);
    }
    (ChunkCheck::Ok, seq, WAL_HDR..WAL_HDR + len, total)
}

/// Server-visible manifest header: `n(u32) | ids(32*n) | client blob`.
pub fn encode_manifest(ids: &[Id], blob: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(4 + 32 * ids.len() + blob.len());
    m.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for id in ids {
        m.extend_from_slice(id);
    }
    m.extend_from_slice(blob);
    m
}

pub fn decode_manifest(m: &[u8]) -> Option<(Vec<Id>, &[u8])> {
    if m.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes(m[0..4].try_into().unwrap()) as usize;
    let end = 4usize.checked_add(n.checked_mul(32)?)?;
    if m.len() < end {
        return None;
    }
    let ids = m[4..end]
        .chunks(32)
        .map(|c| c.try_into().unwrap())
        .collect();
    Some((ids, &m[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_roundtrip_and_corruption() {
        let id = [7u8; 32];
        let mut v = Vec::new();
        encode_chunk(&id, b"hello world", &mut v);
        let (c, got, p, t) = parse_chunk(&v);
        assert_eq!(c, ChunkCheck::Ok);
        assert_eq!(got, id);
        assert_eq!(&v[p], b"hello world");
        assert_eq!(t, v.len());
        assert_eq!(parse_chunk(&v[..v.len() - 1]).0, ChunkCheck::Short);
        for bit in [
            0usize,
            5 * 8,
            9 * 8 + 3,
            (CHUNK_HDR + 2) * 8,
            (v.len() - 1) * 8,
        ] {
            let mut w = v.clone();
            w[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(parse_chunk(&w).0, ChunkCheck::Ok, "bit {bit} undetected");
        }
    }

    #[test]
    fn wal_roundtrip_and_corruption() {
        let r = encode_wal(42, b"manifest-bytes");
        let (c, seq, m, t) = parse_wal(&r);
        assert_eq!((c, seq, t), (ChunkCheck::Ok, 42, r.len()));
        assert_eq!(&r[m], b"manifest-bytes");
        for i in 0..r.len() {
            let mut w = r.clone();
            w[i] ^= 0x10;
            assert_ne!(parse_wal(&w).0, ChunkCheck::Ok, "byte {i} undetected");
        }
    }

    #[test]
    fn manifest_header_roundtrip() {
        let ids = vec![[1u8; 32], [2u8; 32]];
        let m = encode_manifest(&ids, b"blob");
        let (got, blob) = decode_manifest(&m).unwrap();
        assert_eq!(got, ids);
        assert_eq!(blob, b"blob");
        assert!(decode_manifest(&m[..10]).is_none());
    }
}
