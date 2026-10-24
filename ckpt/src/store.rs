//! Crash-consistent content-addressed store: log-structured chunk segments, RocksDB index,
//! checksummed manifest WAL, recovery and GC (docs/protocols.md P1).

use crate::format::{self, ChunkCheck, Id};
use crate::io::{self, Backend, IoKind};
use rocksdb::{Options, WriteBatch, WriteOptions, DB};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SEGMENT_MAX: u64 = 256 << 20;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Integrity violation detected (checksum, missing chunk, WAL corruption).
    Corrupt(String),
    NotFound(String),
    Invalid(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<rocksdb::Error> for Error {
    fn from(e: rocksdb::Error) -> Self {
        Error::Io(std::io::Error::other(e.to_string()))
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Corrupt(s) => write!(f, "corrupt: {s}"),
            Error::NotFound(s) => write!(f, "not found: {s}"),
            Error::Invalid(s) => write!(f, "invalid: {s}"),
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loc {
    pub seg: u32,
    pub off: u64,
    pub len: u32,
    pub epoch: u64,
}

impl Loc {
    fn enc(&self) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[0..4].copy_from_slice(&self.seg.to_le_bytes());
        b[4..12].copy_from_slice(&self.off.to_le_bytes());
        b[12..16].copy_from_slice(&self.len.to_le_bytes());
        b[16..24].copy_from_slice(&self.epoch.to_le_bytes());
        b
    }
    fn dec(b: &[u8]) -> Option<Loc> {
        if b.len() != 24 {
            return None;
        }
        Some(Loc {
            seg: u32::from_le_bytes(b[0..4].try_into().ok()?),
            off: u64::from_le_bytes(b[4..12].try_into().ok()?),
            len: u32::from_le_bytes(b[12..16].try_into().ok()?),
            epoch: u64::from_le_bytes(b[16..24].try_into().ok()?),
        })
    }
    fn end(&self) -> u64 {
        self.off + format::CHUNK_OVERHEAD as u64 + self.len as u64
    }
}

fn ckey(id: &Id) -> Vec<u8> {
    let mut k = Vec::with_capacity(33);
    k.push(b'c');
    k.extend_from_slice(id);
    k
}
fn wm_key(seg: u32) -> Vec<u8> {
    let mut k = b"m:seg:".to_vec();
    k.extend_from_slice(&seg.to_be_bytes());
    k
}
const EPOCH_KEY: &[u8] = b"m:epoch";

#[derive(Clone, Debug)]
struct WalEntry {
    seq: u64,
    off: u64,
    total: usize,
}

#[derive(Default, Debug, Clone)]
pub struct Counters {
    pub put_chunks: u64,
    pub put_logical_bytes: u64,
    pub dedup_hits: u64,
    pub dedup_bytes: u64,
    pub segment_bytes_written: u64,
    pub wal_bytes_written: u64,
    pub group_commits: u64,
    pub commits: u64,
    pub gc_runs: u64,
    pub gc_bytes_copied: u64,
    pub gc_bytes_freed: u64,
}

pub struct Store {
    root: PathBuf,
    db: DB,
    io: Box<dyn Backend>,
    segs: BTreeMap<u32, File>,
    seg_len: BTreeMap<u32, u64>,
    active: Option<u32>,
    wal: File,
    wal_len: u64,
    manifests: Vec<WalEntry>,
    /// Set when recovery found corruption in the middle of the WAL: commits are refused and LATEST
    /// reports a detected error instead of silently returning an older checkpoint.
    pub wal_corrupt: Option<String>,
    epoch: u64,
    pub counters: Counters,
    pub recovery: serde_json::Value,
}

fn seg_path(root: &Path, id: u32) -> PathBuf {
    root.join("segments").join(format!("seg-{id:08}.log"))
}

fn fsync_dir(p: &Path) -> std::io::Result<()> {
    File::open(p)?.sync_all()
}

struct ScanResult {
    valid_end: u64,
    records: Vec<(Id, u64, u32)>,
    bad_regions: u64,
}

/// Scan chunk records in `data` starting at `from`. On an invalid record, resynchronise by searching
/// for the next valid record, so valid records after a corrupt one (possibly referenced by committed
/// manifests) are never discarded. `valid_end` is the end of the last valid record.
fn scan_segment(data: &[u8], from: usize) -> ScanResult {
    let mut pos = from;
    let mut valid_end = from as u64;
    let mut records = Vec::new();
    let mut bad_regions = 0;
    let mut in_bad = false;
    while pos < data.len() {
        let (c, id, p, total) = format::parse_chunk(&data[pos..]);
        if c == ChunkCheck::Ok {
            records.push((id, pos as u64, (p.end - p.start) as u32));
            pos += total;
            valid_end = pos as u64;
            in_bad = false;
            continue;
        }
        if !in_bad {
            bad_regions += 1;
            in_bad = true;
        }
        // resync: next occurrence of the magic
        match data[pos + 1..]
            .windows(4)
            .position(|w| w == format::CHUNK_MAGIC)
        {
            Some(k) => pos = pos + 1 + k,
            None => break,
        }
    }
    ScanResult {
        valid_end,
        records,
        bad_regions,
    }
}

impl Store {
    pub fn open(root: &Path, io_kind: IoKind, io_threads: usize) -> Result<Store> {
        let t0 = Instant::now();
        fs::create_dir_all(root.join("segments"))?;
        fs::create_dir_all(root.join("wal"))?;
        // 1. index
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_paranoid_checks(true);
        let idx = root.join("index");
        let mut index_rebuilt = false;
        let db = match DB::open(&opts, &idx) {
            Ok(db) => db,
            Err(e) => {
                // The index is derivable from segments: quarantine and rebuild.
                eprintln!("index open failed ({e}); rebuilding from segments");
                let q = root.join(format!(
                    "index.corrupt-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                fs::rename(&idx, &q)?;
                index_rebuilt = true;
                DB::open(&opts, &idx)?
            }
        };
        let epoch = db
            .get(EPOCH_KEY)?
            .and_then(|v| v.try_into().ok().map(u64::from_le_bytes))
            .unwrap_or(1);

        // 2. segments: scan unindexed tails (or everything after an index rebuild)
        let mut seg_ids: Vec<u32> = fs::read_dir(root.join("segments"))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().into_string().ok()?;
                n.strip_prefix("seg-")?.strip_suffix(".log")?.parse().ok()
            })
            .collect();
        seg_ids.sort();
        let mut segs = BTreeMap::new();
        let mut seg_len = BTreeMap::new();
        let mut scanned: HashMap<Id, Loc> = HashMap::new();
        let mut batch = WriteBatch::default();
        let mut stats_scanned_bytes = 0u64;
        let mut stats_bad_regions = 0u64;
        let mut stats_truncated = 0u64;
        for &sid in &seg_ids {
            let p = seg_path(root, sid);
            let f = OpenOptions::new().read(true).write(true).open(&p)?;
            let flen = f.metadata()?.len();
            let wm = if index_rebuilt {
                0
            } else {
                db.get(wm_key(sid))?
                    .and_then(|v| v.try_into().ok().map(u64::from_le_bytes))
                    .unwrap_or(0)
            };
            let from = if wm > flen { 0 } else { wm };
            let mut data = Vec::with_capacity((flen - from) as usize);
            let mut rf = &f;
            std::io::Seek::seek(&mut rf, std::io::SeekFrom::Start(from))?;
            rf.read_to_end(&mut data)?;
            stats_scanned_bytes += data.len() as u64;
            let sr = scan_segment(&data, 0);
            stats_bad_regions += sr.bad_regions;
            let valid_end = from + sr.valid_end;
            for (id, off, len) in sr.records {
                scanned.insert(
                    id,
                    Loc {
                        seg: sid,
                        off: from + off,
                        len,
                        epoch,
                    },
                );
            }
            if flen > valid_end {
                // torn / garbage tail after the last valid record
                f.set_len(valid_end)?;
                f.sync_all()?;
                stats_truncated += flen - valid_end;
            }
            batch.put(wm_key(sid), valid_end.to_le_bytes());
            seg_len.insert(sid, valid_end);
            segs.insert(sid, f);
        }

        // 3. index cross-check: drop entries that point past a segment's valid end; repoint to a
        //    scanned copy where one exists; add scanned records that are missing.
        let mut dropped = 0u64;
        let mut repointed = 0u64;
        let mut added = 0u64;
        let mut present: HashSet<Id> = HashSet::new();
        for item in db.prefix_iterator(b"c") {
            let (k, v) = item?;
            if k.first() != Some(&b'c') || k.len() != 33 {
                break;
            }
            let id: Id = k[1..].try_into().unwrap();
            let ok = Loc::dec(&v)
                .map(|l| seg_len.get(&l.seg).is_some_and(|&e| l.end() <= e))
                .unwrap_or(false);
            if ok {
                present.insert(id);
            } else if let Some(l) = scanned.get(&id) {
                batch.put(&k, l.enc());
                present.insert(id);
                repointed += 1;
            } else {
                batch.delete(&k);
                dropped += 1;
            }
        }
        for (id, l) in &scanned {
            if !present.contains(id) {
                batch.put(ckey(id), l.enc());
                added += 1;
            }
        }
        let mut wo = WriteOptions::default();
        wo.set_sync(true);
        db.write_opt(batch, &wo)?;

        // 4. WAL
        let wal_path = root.join("wal").join("manifest.wal");
        let wal = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wal_path)?;
        fsync_dir(&root.join("wal"))?;
        let mut wdata = Vec::new();
        (&wal).read_to_end(&mut wdata)?;
        let mut pos = 0usize;
        let mut manifests = Vec::new();
        let mut wal_corrupt = None;
        let mut last_seq = 0u64;
        while pos < wdata.len() {
            let (c, seq, _m, total) = format::parse_wal(&wdata[pos..]);
            if c == ChunkCheck::Ok && seq > last_seq {
                manifests.push(WalEntry {
                    seq,
                    off: pos as u64,
                    total,
                });
                last_seq = seq;
                pos += total;
                continue;
            }
            // invalid record: torn tail iff no valid record follows
            let mut later = false;
            let mut q = pos + 1;
            while let Some(k) = wdata[q..].windows(4).position(|w| w == format::WAL_MAGIC) {
                let at = q + k;
                if format::parse_wal(&wdata[at..]).0 == ChunkCheck::Ok {
                    later = true;
                    break;
                }
                q = at + 1;
            }
            if later || (c == ChunkCheck::Ok && seq <= last_seq) {
                wal_corrupt = Some(format!(
                    "WAL record at offset {pos} invalid but later records exist"
                ));
            }
            break;
        }
        let wal_truncated = if wal_corrupt.is_none() && (pos as u64) < wdata.len() as u64 {
            let t = wdata.len() as u64 - pos as u64;
            wal.set_len(pos as u64)?;
            wal.sync_all()?;
            t
        } else {
            0
        };
        let wal_len = if wal_corrupt.is_some() {
            wdata.len() as u64
        } else {
            pos as u64
        };

        let recovery = json!({
            "recovery_ms": t0.elapsed().as_secs_f64() * 1e3,
            "latest_seq": manifests.last().map(|m: &WalEntry| m.seq),
            "manifests": manifests.len(),
            "segments": seg_ids.len(),
            "segment_bytes_scanned": stats_scanned_bytes,
            "segment_bad_regions": stats_bad_regions,
            "segment_bytes_truncated": stats_truncated,
            "index_rebuilt": index_rebuilt,
            "index_dropped": dropped,
            "index_repointed": repointed,
            "index_added": added,
            "wal_truncated_bytes": wal_truncated,
            "wal_corrupt": wal_corrupt.clone(),
            "io": io_kind.name(),
        });
        let backend = io::make(io_kind, io_threads);
        Ok(Store {
            root: root.to_path_buf(),
            db,
            io: backend,
            segs,
            seg_len,
            active: None,
            wal,
            wal_len,
            manifests,
            wal_corrupt,
            epoch,
            counters: Counters::default(),
            recovery,
        })
    }

    pub fn io_kind(&self) -> IoKind {
        self.io.kind()
    }

    fn lookup(&self, id: &Id) -> Result<Option<Loc>> {
        Ok(self.db.get(ckey(id))?.and_then(|v| Loc::dec(&v)))
    }

    fn new_segment(&mut self) -> Result<u32> {
        let sid = self.segs.keys().next_back().map(|s| s + 1).unwrap_or(1);
        let p = seg_path(&self.root, sid);
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&p)?;
        fsync_dir(&self.root.join("segments"))?;
        self.segs.insert(sid, f);
        self.seg_len.insert(sid, 0);
        self.active = Some(sid);
        Ok(sid)
    }

    /// Append `bufs` (already-encoded records with their offsets relative to buffer start) durably to
    /// the active segment; returns (segment, base offset).
    fn append_durable(&mut self, buf: &[u8]) -> Result<(u32, u64)> {
        let sid = match self.active {
            Some(s)
                if self.seg_len[&s] + buf.len() as u64 <= SEGMENT_MAX || self.seg_len[&s] == 0 =>
            {
                s
            }
            _ => self.new_segment()?,
        };
        let base = self.seg_len[&sid];
        let f = &self.segs[&sid];
        self.io.write_durable(f, base, buf)?;
        self.seg_len.insert(sid, base + buf.len() as u64);
        self.counters.segment_bytes_written += buf.len() as u64;
        Ok((sid, base))
    }

    /// Group commit of many chunks: one durable append, then index insert. Returns per-chunk dedup flag.
    pub fn put_chunks(&mut self, chunks: &[(Id, &[u8])]) -> Result<Vec<bool>> {
        let mut dedup = vec![false; chunks.len()];
        let mut buf = Vec::new();
        let mut pending: Vec<(Id, u64, u32)> = Vec::new();
        let mut seen: HashSet<Id> = HashSet::new();
        let mut touch = WriteBatch::default();
        for (i, (id, payload)) in chunks.iter().enumerate() {
            if payload.len() > format::MAX_CHUNK {
                return Err(Error::Invalid("chunk too large".into()));
            }
            self.counters.put_chunks += 1;
            self.counters.put_logical_bytes += payload.len() as u64;
            if seen.contains(id) {
                dedup[i] = true;
                self.counters.dedup_hits += 1;
                self.counters.dedup_bytes += payload.len() as u64;
                continue;
            }
            if let Some(mut l) = self.lookup(id)? {
                dedup[i] = true;
                self.counters.dedup_hits += 1;
                self.counters.dedup_bytes += payload.len() as u64;
                if l.epoch < self.epoch {
                    l.epoch = self.epoch;
                    touch.put(ckey(id), l.enc());
                }
                continue;
            }
            seen.insert(*id);
            let off = buf.len() as u64;
            format::encode_chunk(id, payload, &mut buf);
            pending.push((*id, off, payload.len() as u32));
        }
        let mut batch = touch;
        if !buf.is_empty() {
            let (sid, base) = self.append_durable(&buf)?;
            for (id, off, len) in pending {
                let l = Loc {
                    seg: sid,
                    off: base + off,
                    len,
                    epoch: self.epoch,
                };
                batch.put(ckey(&id), l.enc());
            }
            batch.put(wm_key(sid), self.seg_len[&sid].to_le_bytes());
            self.counters.group_commits += 1;
        }
        // Index entries are inserted only after the segment bytes are fsync'd (append_durable).
        self.db.write(batch)?;
        Ok(dedup)
    }

    /// Which ids are durably present. Hits are re-stamped with the current epoch so a concurrent GC
    /// does not collect a chunk a client is about to reference without re-uploading.
    pub fn has_chunks(&mut self, ids: &[Id]) -> Result<Vec<bool>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut b = WriteBatch::default();
        for id in ids {
            match self.lookup(id)? {
                Some(mut l) => {
                    if l.epoch < self.epoch {
                        l.epoch = self.epoch;
                        b.put(ckey(id), l.enc());
                    }
                    out.push(true)
                }
                None => out.push(false),
            }
        }
        self.db.write(b)?;
        Ok(out)
    }

    pub fn latest(&self) -> Result<Option<u64>> {
        if let Some(e) = &self.wal_corrupt {
            return Err(Error::Corrupt(e.clone()));
        }
        Ok(self.manifests.last().map(|m| m.seq))
    }

    pub fn list(&self) -> Vec<u64> {
        self.manifests.iter().map(|m| m.seq).collect()
    }

    pub fn commit(&mut self, seq: u64, ids: &[Id], blob: &[u8]) -> Result<()> {
        if let Some(e) = &self.wal_corrupt {
            return Err(Error::Corrupt(e.clone()));
        }
        if let Some(last) = self.manifests.last() {
            if seq <= last.seq {
                return Err(Error::Invalid(format!(
                    "non-monotonic seq {seq} <= {}",
                    last.seq
                )));
            }
        }
        if seq == 0 {
            return Err(Error::Invalid("seq must be > 0".into()));
        }
        let mut missing = 0usize;
        // Refresh the epoch of *every* referenced chunk (deduplicated ones included) before the WAL
        // append, so a GC whose mark predates this commit re-checks and retains them at sweep time
        // (TLA+ finding, docs/protocols.md P1 step 4).
        let mut refresh = WriteBatch::default();
        for id in ids {
            match self.lookup(id)? {
                Some(mut l) if self.seg_len.get(&l.seg).is_some_and(|&e| l.end() <= e) => {
                    if l.epoch < self.epoch {
                        l.epoch = self.epoch;
                        refresh.put(ckey(id), l.enc());
                    }
                }
                _ => missing += 1,
            }
        }
        if missing > 0 {
            return Err(Error::NotFound(format!(
                "{missing} referenced chunks not durable"
            )));
        }
        self.db.write(refresh)?;
        let rec = format::encode_wal(seq, &format::encode_manifest(ids, blob));
        let appended = self
            .wal
            .write_all_at(&rec, self.wal_len)
            .and_then(|_| self.wal.sync_data());
        if let Err(e) = appended {
            // Not acknowledged: try to remove the partial/unsynced record so it cannot resurface
            // after recovery. If even that fails, GC still treats any WAL tail as live (see gc).
            let _ = self.wal.set_len(self.wal_len);
            let _ = self.wal.sync_data();
            return Err(e.into());
        }
        self.manifests.push(WalEntry {
            seq,
            off: self.wal_len,
            total: rec.len(),
        });
        self.wal_len += rec.len() as u64;
        self.counters.wal_bytes_written += rec.len() as u64;
        self.counters.commits += 1;
        Ok(())
    }

    fn read_wal_entry(&self, e: &WalEntry) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; e.total];
        self.wal.read_exact_at(&mut buf, e.off)?;
        let (c, seq, m, _) = format::parse_wal(&buf);
        if c != ChunkCheck::Ok || seq != e.seq {
            return Err(Error::Corrupt(format!(
                "WAL record seq {} failed checksum",
                e.seq
            )));
        }
        Ok(buf[m].to_vec())
    }

    /// Returns (seq, ids, client blob). `seq == 0` means latest.
    pub fn get_manifest(&self, seq: u64) -> Result<(u64, Vec<Id>, Vec<u8>)> {
        let e = if seq == 0 {
            self.latest()?;
            self.manifests.last()
        } else {
            self.manifests.iter().find(|m| m.seq == seq)
        }
        .ok_or_else(|| Error::NotFound(format!("manifest {seq}")))?
        .clone();
        let m = self.read_wal_entry(&e)?;
        let (ids, blob) =
            format::decode_manifest(&m).ok_or_else(|| Error::Corrupt("manifest header".into()))?;
        Ok((e.seq, ids, blob.to_vec()))
    }

    /// Read and fully re-verify a chunk (magic, crc, payload hash, id).
    pub fn get_chunk(&self, id: &Id) -> Result<Vec<u8>> {
        let l = self
            .lookup(id)?
            .ok_or_else(|| Error::NotFound(format!("chunk {}", hex(id))))?;
        let f = self
            .segs
            .get(&l.seg)
            .ok_or_else(|| Error::Corrupt(format!("segment {} missing", l.seg)))?;
        let mut buf = vec![0u8; format::CHUNK_OVERHEAD + l.len as usize];
        f.read_exact_at(&mut buf, l.off)
            .map_err(|e| Error::Corrupt(format!("chunk {} unreadable: {e}", hex(id))))?;
        let (c, got, p, _) = format::parse_chunk(&buf);
        if c != ChunkCheck::Ok {
            return Err(Error::Corrupt(format!("chunk {}: {:?}", hex(id), c)));
        }
        if &got != id {
            return Err(Error::Corrupt(format!("chunk {}: id mismatch", hex(id))));
        }
        Ok(buf[p].to_vec())
    }

    /// Server-side verification of a manifest: every chunk present and intact.
    pub fn verify(&self, seq: u64) -> Result<serde_json::Value> {
        let (seq, ids, _) = self.get_manifest(seq)?;
        let mut missing = 0;
        let mut corrupt = 0;
        for id in &ids {
            match self.get_chunk(id) {
                Ok(_) => {}
                Err(Error::NotFound(_)) => missing += 1,
                Err(_) => corrupt += 1,
            }
        }
        Ok(
            json!({"seq": seq, "chunks": ids.len(), "missing": missing, "corrupt": corrupt,
                  "ok": missing == 0 && corrupt == 0}),
        )
    }

    /// Garbage collection (P1 step 4). Retain the last `keep` manifests.
    pub fn gc(&mut self, keep: usize) -> Result<serde_json::Value> {
        self.gc_with(keep, &mut |_| Ok(()))
    }

    /// Ids referenced by WAL records that sit in the file beyond the acknowledged length (appended but
    /// never fsync'd/acked, e.g. after a failed fsync). They might still surface after a crash, so GC
    /// treats their chunks as live.
    fn wal_tail_ids(&self) -> Result<Vec<Id>> {
        let flen = self.wal.metadata()?.len();
        let mut out = Vec::new();
        if flen <= self.wal_len {
            return Ok(out);
        }
        let mut buf = vec![0u8; (flen - self.wal_len) as usize];
        self.wal.read_exact_at(&mut buf, self.wal_len)?;
        let mut pos = 0;
        while pos < buf.len() {
            let (c, _, m, total) = format::parse_wal(&buf[pos..]);
            if c != ChunkCheck::Ok {
                break;
            }
            if let Some((ids, _)) = format::decode_manifest(&buf[pos..][m]) {
                out.extend(ids);
            }
            pos += total;
        }
        Ok(out)
    }

    /// Epoch stamped on a chunk's index entry (tests / diagnostics).
    pub fn chunk_epoch(&self, id: &Id) -> Option<u64> {
        self.lookup(id).ok().flatten().map(|l| l.epoch)
    }

    /// GC with a hook run between mark and sweep. In the server, GC and COMMIT are serialised by the
    /// writer thread, so the hook is a no-op; tests use it to inject a COMMIT inside the window the
    /// TLA+ model found, checking that the sweep-time epoch re-check keeps those chunks.
    pub fn gc_with(
        &mut self,
        keep: usize,
        between_mark_and_sweep: &mut dyn FnMut(&mut Store) -> Result<()>,
    ) -> Result<serde_json::Value> {
        if let Some(e) = &self.wal_corrupt {
            return Err(Error::Corrupt(e.clone()));
        }
        let t0 = Instant::now();
        let gc_epoch = self.epoch;
        self.epoch += 1;
        // seal the active segment so it can be compacted too; new appends go to a fresh segment
        self.active = None;
        self.db.put(EPOCH_KEY, self.epoch.to_le_bytes())?;
        let keep = keep.max(1);
        let cut = self.manifests.len().saturating_sub(keep);
        let retained: Vec<WalEntry> = self.manifests[cut..].to_vec();

        // (a) rewrite WAL with retained manifests (atomic rename)
        let mut live: HashSet<Id> = HashSet::new();
        let tail = self.wal_tail_ids()?;
        let tail_live = tail.len();
        live.extend(tail);
        let mut neww = Vec::new();
        let mut new_entries = Vec::new();
        for e in &retained {
            let m = self.read_wal_entry(e)?;
            let (ids, _) =
                format::decode_manifest(&m).ok_or_else(|| Error::Corrupt("manifest".into()))?;
            live.extend(ids);
            let rec = format::encode_wal(e.seq, &m);
            new_entries.push(WalEntry {
                seq: e.seq,
                off: neww.len() as u64,
                total: rec.len(),
            });
            neww.extend_from_slice(&rec);
        }
        let dropped_manifests = cut;
        if cut > 0 {
            let wdir = self.root.join("wal");
            let tmp = wdir.join("manifest.wal.new");
            {
                let f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&tmp)?;
                f.write_all_at(&neww, 0)?;
                f.sync_all()?;
            }
            fs::rename(&tmp, wdir.join("manifest.wal"))?;
            fsync_dir(&wdir)?;
            self.wal = OpenOptions::new()
                .read(true)
                .write(true)
                .open(wdir.join("manifest.wal"))?;
            self.wal_len = neww.len() as u64;
            self.manifests = new_entries;
        }

        // (b) classify index entries
        let mut per_seg_live: BTreeMap<u32, Vec<(Id, Loc)>> = BTreeMap::new();
        let mut per_seg_dead: BTreeMap<u32, u64> = BTreeMap::new();
        let mut dead_ids: Vec<Id> = Vec::new();
        for item in self.db.prefix_iterator(b"c") {
            let (k, v) = item?;
            if k.first() != Some(&b'c') || k.len() != 33 {
                break;
            }
            let id: Id = k[1..].try_into().unwrap();
            let Some(l) = Loc::dec(&v) else { continue };
            if live.contains(&id) || l.epoch >= gc_epoch {
                per_seg_live.entry(l.seg).or_default().push((id, l));
            } else {
                *per_seg_dead.entry(l.seg).or_default() += l.end() - l.off;
                dead_ids.push(id);
            }
        }
        let mark_segs: Vec<u32> = self.segs.keys().copied().collect();

        between_mark_and_sweep(self)?;

        // Sweep-time re-check: a chunk marked dead whose epoch has since been refreshed to >= the GC
        // start epoch (by a COMMIT, HAS_CHUNKS or PUT that raced the mark) is retained.
        let mut rescued = 0u64;
        let mut still_dead = Vec::with_capacity(dead_ids.len());
        per_seg_dead.clear();
        for id in dead_ids {
            match self.lookup(&id)? {
                Some(l) if l.epoch >= gc_epoch => {
                    per_seg_live.entry(l.seg).or_default().push((id, l));
                    rescued += 1;
                }
                Some(l) => {
                    *per_seg_dead.entry(l.seg).or_default() += l.end() - l.off;
                    still_dead.push(id);
                }
                None => {}
            }
        }
        let dead_ids = still_dead;
        // victims: segments that existed at mark time (never ones created since, e.g. by puts in the
        // window) with any dead bytes, or with no live chunks at all
        let victims: Vec<u32> = mark_segs
            .iter()
            .copied()
            .filter(|s| Some(*s) != self.active)
            .filter(|s| per_seg_dead.contains_key(s) || !per_seg_live.contains_key(s))
            .collect();
        // (c) copy live chunks out of victims into a new durable segment
        let mut copy: Vec<(Id, Vec<u8>, u64)> = Vec::new();
        for s in &victims {
            if let Some(v) = per_seg_live.get(s) {
                for (id, l) in v {
                    let p = self.get_chunk(id)?;
                    copy.push((*id, p, l.epoch));
                }
            }
        }
        let mut batch = WriteBatch::default();
        let mut copied = 0u64;
        if !copy.is_empty() {
            let mut buf = Vec::new();
            let mut offs = Vec::new();
            for (id, p, _) in &copy {
                offs.push(buf.len() as u64);
                format::encode_chunk(id, p, &mut buf);
            }
            self.active = None; // fresh segment for compacted data
            let (sid, base) = self.append_durable(&buf)?;
            for ((id, p, ep), o) in copy.iter().zip(offs) {
                let l = Loc {
                    seg: sid,
                    off: base + o,
                    len: p.len() as u32,
                    epoch: *ep,
                };
                batch.put(ckey(id), l.enc());
            }
            batch.put(wm_key(sid), self.seg_len[&sid].to_le_bytes());
            copied = buf.len() as u64;
            self.active = None;
        }
        for id in &dead_ids {
            batch.delete(ckey(id));
        }
        for s in &victims {
            batch.delete(wm_key(*s));
        }
        // (d) index update durable *before* unlinking victims
        let mut wo = WriteOptions::default();
        wo.set_sync(true);
        self.db.write_opt(batch, &wo)?;
        let mut freed = 0u64;
        for s in &victims {
            freed += self.seg_len.get(s).copied().unwrap_or(0);
            self.segs.remove(s);
            self.seg_len.remove(s);
            fs::remove_file(seg_path(&self.root, *s))?;
        }
        fsync_dir(&self.root.join("segments"))?;
        self.counters.gc_runs += 1;
        self.counters.gc_bytes_copied += copied;
        self.counters.gc_bytes_freed += freed;
        Ok(json!({
            "gc_ms": t0.elapsed().as_secs_f64() * 1e3,
            "manifests_dropped": dropped_manifests,
            "manifests_retained": self.manifests.len(),
            "dead_chunks": dead_ids.len(),
            "rescued_at_sweep": rescued,
            "wal_tail_live_refs": tail_live,
            "victim_segments": victims.len(),
            "bytes_copied": copied,
            "bytes_freed": freed,
        }))
    }

    pub fn stats(&self) -> serde_json::Value {
        let c = &self.counters;
        let seg_bytes: u64 = self.seg_len.values().sum();
        json!({
            "io": self.io.kind().name(),
            "epoch": self.epoch,
            "segments": self.segs.len(),
            "segment_bytes": seg_bytes,
            "wal_bytes": self.wal_len,
            "manifests": self.manifests.len(),
            "latest_seq": self.manifests.last().map(|m| m.seq),
            "put_chunks": c.put_chunks,
            "put_logical_bytes": c.put_logical_bytes,
            "dedup_hits": c.dedup_hits,
            "dedup_bytes": c.dedup_bytes,
            "segment_bytes_written": c.segment_bytes_written,
            "wal_bytes_written": c.wal_bytes_written,
            "group_commits": c.group_commits,
            "commits": c.commits,
            "gc_runs": c.gc_runs,
            "gc_bytes_copied": c.gc_bytes_copied,
            "gc_bytes_freed": c.gc_bytes_freed,
            "wal_corrupt": self.wal_corrupt,
            "recovery": self.recovery,
        })
    }
}

pub fn hex(id: &[u8]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u8) -> Id {
        *blake3::hash(&[n]).as_bytes()
    }

    fn open(p: &Path) -> Store {
        Store::open(p, IoKind::Pwrite, 2).unwrap()
    }

    #[test]
    fn put_commit_restore_dedup() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        let dd = s
            .put_chunks(&[(id(1), b"aaa"), (id(2), b"bbb"), (id(1), b"aaa")])
            .unwrap();
        assert_eq!(dd, vec![false, false, true]);
        assert_eq!(s.put_chunks(&[(id(2), b"bbb")]).unwrap(), vec![true]);
        s.commit(1, &[id(1), id(2)], b"blob1").unwrap();
        assert!(
            s.commit(1, &[id(1)], b"x").is_err(),
            "non-monotonic seq accepted"
        );
        assert!(matches!(
            s.commit(2, &[id(9)], b"x"),
            Err(Error::NotFound(_))
        ));
        drop(s);
        let s = open(d.path());
        assert_eq!(s.latest().unwrap(), Some(1));
        let (seq, ids, blob) = s.get_manifest(0).unwrap();
        assert_eq!((seq, ids.len(), blob.as_slice()), (1, 2, &b"blob1"[..]));
        assert_eq!(s.get_chunk(&id(2)).unwrap(), b"bbb");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_backend_roundtrip() {
        let d = tempfile::tempdir().unwrap();
        let mut s = Store::open(d.path(), IoKind::Uring, 2).unwrap();
        // Docker's default seccomp profile blocks io_uring_setup; the lab runs --privileged.
        if s.io_kind() != IoKind::Uring {
            assert!(
                std::env::var("AEGIS_REQUIRE_URING").is_err(),
                "io_uring required but unavailable"
            );
            eprintln!("io_uring unavailable here; skipping uring assertions");
        }
        let big: Vec<u8> = (0..(5 << 20) + 123).map(|i| (i % 251) as u8).collect();
        s.put_chunks(&[(id(1), &big), (id(2), b"small")]).unwrap();
        s.commit(1, &[id(1), id(2)], b"m").unwrap();
        drop(s);
        let s = Store::open(d.path(), IoKind::Uring, 2).unwrap();
        assert_eq!(s.get_chunk(&id(1)).unwrap(), big);
        assert!(s.verify(1).unwrap()["ok"].as_bool().unwrap());
    }

    #[test]
    fn wal_torn_tail_truncated() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        s.put_chunks(&[(id(1), b"x")]).unwrap();
        s.commit(1, &[id(1)], b"m1").unwrap();
        s.commit(2, &[id(1)], b"m2").unwrap();
        drop(s);
        let w = d.path().join("wal/manifest.wal");
        let len = fs::metadata(&w).unwrap().len();
        let f = OpenOptions::new().write(true).open(&w).unwrap();
        f.set_len(len - 5).unwrap(); // tear the last record
        drop(f);
        let s = open(d.path());
        assert_eq!(s.latest().unwrap(), Some(1));
        assert!(s.recovery["wal_truncated_bytes"].as_u64().unwrap() > 0);
        assert!(s.wal_corrupt.is_none());
    }

    #[test]
    fn wal_mid_corruption_detected_not_truncated() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        s.put_chunks(&[(id(1), b"x")]).unwrap();
        for q in 1..=3 {
            s.commit(q, &[id(1)], b"mmmm").unwrap();
        }
        drop(s);
        let w = d.path().join("wal/manifest.wal");
        let mut data = fs::read(&w).unwrap();
        data[20] ^= 0x01; // inside record 1
        fs::write(&w, &data).unwrap();
        let s = open(d.path());
        assert!(matches!(s.latest(), Err(Error::Corrupt(_))));
        assert_eq!(fs::metadata(&w).unwrap().len(), data.len() as u64);
    }

    #[test]
    fn index_cross_check_after_segment_truncation() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        s.put_chunks(&[(id(1), b"first")]).unwrap();
        s.put_chunks(&[(id(2), b"second-chunk")]).unwrap();
        s.commit(1, &[id(1), id(2)], b"m").unwrap();
        drop(s);
        let seg = seg_path(d.path(), 1);
        let len = fs::metadata(&seg).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&seg)
            .unwrap()
            .set_len(len - 3)
            .unwrap();
        let s = open(d.path());
        assert_eq!(s.recovery["index_dropped"].as_u64().unwrap(), 1);
        assert_eq!(s.get_chunk(&id(1)).unwrap(), b"first");
        assert!(matches!(s.get_chunk(&id(2)), Err(Error::NotFound(_))));
        let v = s.verify(1).unwrap();
        assert_eq!(v["ok"], false);
    }

    #[test]
    fn index_rebuilt_from_segments_when_lost() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        s.put_chunks(&[(id(1), b"one"), (id(2), b"two")]).unwrap();
        s.commit(1, &[id(1), id(2)], b"m").unwrap();
        drop(s);
        // simulate index loss: wipe RocksDB directory
        fs::remove_dir_all(d.path().join("index")).unwrap();
        let s = open(d.path());
        assert_eq!(s.recovery["index_added"].as_u64().unwrap(), 2);
        assert!(s.verify(1).unwrap()["ok"].as_bool().unwrap());
    }

    #[test]
    fn bitflip_in_chunk_detected_on_read() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        s.put_chunks(&[(id(1), b"payload-bytes")]).unwrap();
        s.commit(1, &[id(1)], b"m").unwrap();
        drop(s);
        let seg = seg_path(d.path(), 1);
        let mut data = fs::read(&seg).unwrap();
        data[format::CHUNK_HDR + 3] ^= 0x80;
        fs::write(&seg, &data).unwrap();
        let s = open(d.path());
        assert!(matches!(s.get_chunk(&id(1)), Err(Error::Corrupt(_))));
    }

    /// Two commits, then a GC so that everything written so far is in an epoch older than the next
    /// GC's start epoch. X is referenced only by m1, which the next gc(1) drops.
    fn aged_store(p: &Path) -> Store {
        let mut s = open(p);
        s.put_chunks(&[(id(1), b"chunk-X")]).unwrap();
        s.commit(1, &[id(1)], b"m1").unwrap();
        s.put_chunks(&[(id(2), b"chunk-Y")]).unwrap();
        s.commit(2, &[id(2)], b"m2").unwrap();
        s.gc(1).unwrap(); // gc epoch 1 -> 2; all chunks still epoch 1
        s
    }

    #[test]
    fn commit_refreshes_epochs_of_deduped_chunks() {
        let d = tempfile::tempdir().unwrap();
        let mut s = aged_store(d.path());
        assert_eq!(s.chunk_epoch(&id(1)), Some(1));
        // commit referencing X without re-uploading it (client dedup'd against an earlier HAS)
        s.commit(3, &[id(1)], b"m3").unwrap();
        assert_eq!(
            s.chunk_epoch(&id(1)),
            Some(2),
            "deduped chunk epoch not refreshed"
        );
    }

    #[test]
    fn gc_commit_between_mark_and_sweep_keeps_chunks() {
        // TLA+ counterexample: COMMIT lands between GC mark and sweep, referencing a pre-existing
        // chunk that the mark classified dead. The sweep re-check must retain it (I1/I3).
        let d = tempfile::tempdir().unwrap();
        let mut s = aged_store(d.path());
        let r = s
            .gc_with(1, &mut |st: &mut Store| st.commit(3, &[id(1)], b"m3"))
            .unwrap();
        assert_eq!(r["rescued_at_sweep"].as_u64().unwrap(), 1, "{r}");
        assert_eq!(s.get_chunk(&id(1)).unwrap(), b"chunk-X");
        assert!(s.verify(3).unwrap()["ok"].as_bool().unwrap());
        drop(s);
        let s = open(d.path());
        assert!(s.verify(3).unwrap()["ok"].as_bool().unwrap());
        assert!(s.verify(2).unwrap()["ok"].as_bool().unwrap());
    }

    #[test]
    fn gc_treats_unsynced_wal_tail_as_live() {
        let d = tempfile::tempdir().unwrap();
        let mut s = aged_store(d.path());
        // A WAL record appended beyond the acknowledged length (append happened, ack did not).
        let rec = format::encode_wal(3, &format::encode_manifest(&[id(1)], b"m3"));
        let wal = d.path().join("wal/manifest.wal");
        let len = fs::metadata(&wal).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&wal)
            .unwrap()
            .write_all_at(&rec, len)
            .unwrap();
        let r = s.gc(1).unwrap();
        assert_eq!(r["wal_tail_live_refs"].as_u64().unwrap(), 1, "{r}");
        assert!(
            s.get_chunk(&id(1)).is_ok(),
            "chunk of unsynced WAL tail swept"
        );
        drop(s);
        // If that record survives (it was written), recovery accepts it and it must be complete.
        let s = open(d.path());
        if s.latest().unwrap() == Some(3) {
            assert!(s.verify(3).unwrap()["ok"].as_bool().unwrap());
        }
    }

    #[test]
    fn gc_keeps_retained_and_inflight_chunks() {
        let d = tempfile::tempdir().unwrap();
        let mut s = open(d.path());
        for q in 1..=4u8 {
            s.put_chunks(&[(id(q), &[q; 100])]).unwrap();
            s.put_chunks(&[(id(100), b"shared")]).unwrap();
            s.commit(q as u64, &[id(q), id(100)], b"m").unwrap();
        }
        // first GC: everything is from the current epoch -> nothing collectable yet
        s.gc(2).unwrap();
        assert_eq!(s.list(), vec![3, 4]);
        // in-flight chunk uploaded after the GC epoch bump
        s.put_chunks(&[(id(50), b"inflight")]).unwrap();
        let r = s.gc(2).unwrap();
        assert!(r["dead_chunks"].as_u64().unwrap() >= 2, "{r}");
        for q in [3u8, 4] {
            assert!(s.get_chunk(&id(q)).is_ok());
        }
        assert!(s.get_chunk(&id(100)).is_ok());
        assert!(s.get_chunk(&id(50)).is_ok(), "in-flight chunk collected");
        assert!(s.get_chunk(&id(1)).is_err());
        s.commit(5, &[id(50), id(100)], b"m5").unwrap();
        drop(s);
        let s = open(d.path());
        for q in [3, 4, 5] {
            assert!(s.verify(q).unwrap()["ok"].as_bool().unwrap());
        }
    }
}
