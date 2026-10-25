//! The chunk store: log-structured segments, RocksDB index, manifest WAL, recovery and GC (protocol P1).
//!
//! Locking (all index *relocations/deletions* hold both `meta` and `gq`):
//! * `meta`: WAL file, committed manifests, per-stream sequence numbers. COMMIT holds it across the
//!   index check, the epoch refresh, the WAL append and the WAL fsync, so there is never an accepted
//!   but un-fsync'd WAL record visible to GC mark (TLA+ `FixLivePending` is satisfied trivially).
//! * `gq`: group-commit queue, dedup lookups, index inserts. A leader thread drains the queue, writes
//!   one contiguous run per segment, fdatasyncs, and only then inserts index entries (synchronous
//!   RocksDB write) and wakes the waiters, which then ack (TLA+ `BugIndexBeforeFsync` = FALSE).
//! * `writer`: the active segment (held by the flush leader, and by GC mark when sealing).
//! * `gc_lock`: one GC at a time.

use crate::format::*;
use crate::io::{IoBackend, ReadReq};
use anyhow::{anyhow, bail, ensure, Context, Result};
use rocksdb::{IteratorMode, Options, WriteBatch, WriteOptions, DB};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Config {
    pub root: PathBuf,
    pub io: String,
    pub threads: usize,
    pub seg_max: u64,
}

impl Config {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Config {
            root: root.into(),
            io: "auto".into(),
            threads: 4,
            seg_max: 64 << 20,
        }
    }
}

/// Index value: segment, record offset, payload length, epoch.
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
    fn rec_end(&self) -> u64 {
        self.off + seg_record_len(self.len as usize) as u64
    }
}

#[derive(Clone)]
pub struct ManEntry {
    pub wal_seq: u64,
    pub hdr: ManifestHeader,
    pub bytes: Arc<Vec<u8>>,
}

struct Meta {
    wal: File,
    wal_end: u64,
    next_seq: u64,
    mans: BTreeMap<u64, ManEntry>, // wal_seq -> manifest (retained)
    by_name: HashMap<String, u64>, // name -> wal_seq of newest
    by_stream: HashMap<String, BTreeMap<u64, u64>>, // stream -> mseq -> wal_seq
    wal_error: Option<String>,
}

struct Gq {
    pending: Vec<(ChunkId, Arc<Vec<u8>>)>,
    pending_ids: HashMap<ChunkId, u64>,
    next_ticket: u64,
    done_ticket: u64,
    flushing: bool,
    failed: Option<String>,
}

struct Writer {
    id: u32,
    file: Arc<File>,
    off: u64,
}

#[derive(Default, Serialize)]
pub struct Stats {
    pub logical_bytes: AtomicU64,
    pub dedup_hits: AtomicU64,
    pub dedup_bytes: AtomicU64,
    pub chunks_written: AtomicU64,
    pub seg_bytes_written: AtomicU64,
    pub wal_bytes_written: AtomicU64,
    pub data_fsyncs: AtomicU64,
    pub commits: AtomicU64,
    pub gc_runs: AtomicU64,
    pub gc_bytes_copied: AtomicU64,
    pub gc_segments_deleted: AtomicU64,
    pub gc_chunks_dropped: AtomicU64,
}

#[derive(Default, Serialize, Clone, Debug)]
pub struct RecoveryReport {
    pub open_ms: f64,
    pub index_rebuilt: bool,
    pub index_open_error: Option<String>,
    pub segments: usize,
    pub segment_bytes: u64,
    pub segment_torn_bytes: u64,
    pub index_entries: u64,
    pub index_dropped: u64,
    pub wal_records: u64,
    pub wal_truncated_bytes: u64,
    pub wal_error: Option<String>,
    pub latest_wal_seq: Option<u64>,
    pub latest_missing_chunks: u64,
}

#[derive(Serialize, Debug, Default)]
pub struct PutReport {
    pub dedup: u32,
    pub written: u32,
    /// (segment, end offset) of each record this call wrote, durable at return.
    pub durable: Vec<(u32, u64)>,
}

#[derive(Serialize, Debug)]
pub struct CommitReport {
    pub wal_seq: u64,
    pub wal_end: u64,
}

#[derive(Debug)]
pub enum CommitError {
    Missing(Vec<ChunkId>),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for CommitError {
    fn from(e: anyhow::Error) -> Self {
        CommitError::Other(e)
    }
}

#[derive(Serialize, Debug, Default)]
pub struct GcReport {
    pub start_epoch: u64,
    pub retained_manifests: usize,
    pub dropped_manifests: usize,
    pub live_chunks: usize,
    pub victims: Vec<u32>,
    pub copied_chunks: u64,
    pub copied_bytes: u64,
    pub recheck_copies: u64,
    pub dropped_chunks: u64,
    pub deleted_segments: u64,
    pub reclaimed_bytes: u64,
    pub ms: f64,
}

pub type GcHook = Box<dyn Fn(&str) + Send>;

pub struct Store {
    pub cfg: Config,
    io: Box<dyn IoBackend>,
    db: DB,
    meta: Mutex<Meta>,
    gq: Mutex<Gq>,
    cv: Condvar,
    writer: Mutex<Writer>,
    segs: Mutex<BTreeMap<u32, Arc<File>>>,
    next_seg: AtomicU32,
    epoch: AtomicU64,
    gc_lock: Mutex<()>,
    pub stats: Stats,
    pub recovery: RecoveryReport,
    /// Test hook: called between GC phases (mark->copy, copy->sweep) to force interleavings.
    pub gc_hook: Mutex<Option<GcHook>>,
}

fn seg_path(root: &Path, id: u32) -> PathBuf {
    root.join("segments").join(format!("seg-{id:08}.log"))
}

fn sync_dir(p: &Path) -> Result<()> {
    File::open(p)?.sync_all()?;
    Ok(())
}

fn wopts(sync: bool) -> WriteOptions {
    let mut w = WriteOptions::default();
    w.set_sync(sync);
    w
}

/// Scan a segment's record headers; returns the end of the longest valid prefix and its records.
fn scan_segment(f: &File, len: u64, mut each: impl FnMut(SegHeader, u64)) -> u64 {
    let mut off = 0u64;
    let mut hdr = [0u8; SEG_HDR];
    let mut tr = [0u8; SEG_TRAILER];
    while off + (SEG_HDR + SEG_TRAILER) as u64 <= len {
        if f.read_exact_at(&mut hdr, off).is_err() {
            break;
        }
        let Some(h) = parse_seg_header(&hdr) else {
            break;
        };
        let end = off + seg_record_len(h.len as usize) as u64;
        if end > len
            || f.read_exact_at(&mut tr, end - SEG_TRAILER as u64).is_err()
            || !check_seg_crc(&hdr, &tr)
        {
            break;
        }
        each(h, off);
        off = end;
    }
    off
}

impl Store {
    pub fn open(cfg: Config) -> Result<Store> {
        let t0 = Instant::now();
        let io = crate::io::make(&cfg.io, cfg.threads)?;
        let root = cfg.root.clone();
        for d in ["segments", "wal"] {
            fs::create_dir_all(root.join(d))?;
        }
        let mut rep = RecoveryReport::default();

        // --- index -------------------------------------------------------------------------
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_paranoid_checks(true);
        opts.set_max_background_jobs(2);
        let ipath = root.join("index");
        let db = match DB::open(&opts, &ipath) {
            Ok(db) => db,
            Err(e) => {
                // The index is derived data: segments are self-describing, so rebuild it.
                rep.index_open_error = Some(e.to_string());
                rep.index_rebuilt = true;
                let aside = root.join(format!("index.corrupt.{}", std::process::id()));
                let _ = fs::remove_dir_all(&aside);
                fs::rename(&ipath, &aside).context("moving corrupt index aside")?;
                DB::open(&opts, &ipath)?
            }
        };

        // --- segments ------------------------------------------------------------------------
        let mut segs = BTreeMap::new();
        let mut valid_end: HashMap<u32, u64> = HashMap::new();
        let mut rebuild = WriteBatch::default();
        for ent in fs::read_dir(root.join("segments"))? {
            let ent = ent?;
            let name = ent.file_name().to_string_lossy().to_string();
            let Some(id) = name
                .strip_prefix("seg-")
                .and_then(|s| s.strip_suffix(".log"))
            else {
                continue;
            };
            let id: u32 = id.parse().context("segment name")?;
            let f = OpenOptions::new().read(true).write(true).open(ent.path())?;
            let len = f.metadata()?.len();
            let rebuilding = rep.index_rebuilt;
            let end = scan_segment(&f, len, |h, off| {
                if rebuilding {
                    let l = Loc {
                        seg: id,
                        off,
                        len: h.len,
                        epoch: 0,
                    };
                    rebuild.put(h.id, l.enc());
                }
            });
            rep.segment_bytes += len;
            rep.segment_torn_bytes += len - end;
            valid_end.insert(id, end);
            segs.insert(id, Arc::new(f));
        }
        rep.segments = segs.len();
        if rep.index_rebuilt {
            db.write_opt(rebuild, &wopts(true))?;
        }

        // --- index cross-check against segments ---------------------------------------------
        let mut drop = WriteBatch::default();
        let mut max_epoch = 0u64;
        let mut hdr = [0u8; SEG_HDR];
        for kv in db.iterator(IteratorMode::Start) {
            let (k, v) = kv?;
            rep.index_entries += 1;
            let ok = (|| {
                let l = Loc::dec(&v)?;
                let f = segs.get(&l.seg)?;
                if l.rec_end() > *valid_end.get(&l.seg)? || k.len() != 32 {
                    return None;
                }
                f.read_exact_at(&mut hdr, l.off).ok()?;
                let h = parse_seg_header(&hdr)?;
                (h.id[..] == k[..] && h.len == l.len).then_some(l)
            })();
            match ok {
                Some(l) => max_epoch = max_epoch.max(l.epoch),
                None => {
                    rep.index_dropped += 1;
                    drop.delete(&k);
                }
            }
        }
        if rep.index_dropped > 0 {
            db.write_opt(drop, &wopts(true))?;
        }

        // --- WAL -----------------------------------------------------------------------------
        let wpath = root.join("wal").join("manifest.wal");
        let wal = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wpath)?;
        let buf = fs::read(&wpath)?;
        let mut meta = Meta {
            wal: wal.try_clone()?,
            wal_end: 0,
            next_seq: 1,
            mans: BTreeMap::new(),
            by_name: HashMap::new(),
            by_stream: HashMap::new(),
            wal_error: None,
        };
        let mut pos = 0usize;
        let mut last_seq = 0u64;
        while pos < buf.len() {
            match parse_wal_record(&buf[pos..]) {
                Some((seq, range, total)) => {
                    let mb = &buf[pos + range.start..pos + range.end];
                    if seq <= last_seq {
                        meta.wal_error = Some(format!("WAL sequence regression at offset {pos}"));
                        break;
                    }
                    match parse_manifest(mb) {
                        Ok(h) => Self::register(&mut meta, seq, h, Arc::new(mb.to_vec())),
                        Err(e) => {
                            meta.wal_error =
                                Some(format!("unparseable manifest at offset {pos}: {e}"));
                            break;
                        }
                    }
                    last_seq = seq;
                    rep.wal_records += 1;
                    pos += total;
                }
                None => {
                    // Torn tail, or corruption in the middle? Any valid record after `pos` means the
                    // latter: refuse to silently drop committed checkpoints.
                    let later = (pos + 1..buf.len()).find(|&p| {
                        buf[p..].starts_with(WAL_MAGIC) && parse_wal_record(&buf[p..]).is_some()
                    });
                    if let Some(p) = later {
                        meta.wal_error =
                            Some(format!("WAL record at offset {pos} is corrupt but a valid record follows at {p}"));
                    } else {
                        rep.wal_truncated_bytes = (buf.len() - pos) as u64;
                        wal.set_len(pos as u64)?;
                        wal.sync_all()?;
                    }
                    break;
                }
            }
        }
        meta.wal_end = pos as u64;
        meta.next_seq = last_seq + 1;
        rep.wal_error = meta.wal_error.clone();
        rep.latest_wal_seq = meta.mans.keys().next_back().copied();
        if let Some(m) = meta.mans.values().next_back() {
            rep.latest_missing_chunks = m
                .hdr
                .ids
                .iter()
                .filter(|id| db.get(id).ok().flatten().is_none())
                .count() as u64;
        }

        // --- fresh active segment (never append to a recovered one) --------------------------
        let next = segs.keys().next_back().map(|x| x + 1).unwrap_or(1);
        let f = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(seg_path(&root, next))?,
        );
        sync_dir(&root.join("segments"))?;
        segs.insert(next, f.clone());
        rep.open_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Store {
            cfg,
            io,
            db,
            meta: Mutex::new(meta),
            gq: Mutex::new(Gq {
                pending: Vec::new(),
                pending_ids: HashMap::new(),
                next_ticket: 1,
                done_ticket: 0,
                flushing: false,
                failed: None,
            }),
            cv: Condvar::new(),
            writer: Mutex::new(Writer {
                id: next,
                file: f,
                off: 0,
            }),
            segs: Mutex::new(segs),
            next_seg: AtomicU32::new(next + 1),
            epoch: AtomicU64::new(max_epoch + 1),
            gc_lock: Mutex::new(()),
            stats: Stats::default(),
            recovery: rep,
            gc_hook: Mutex::new(None),
        })
    }

    pub fn io_name(&self) -> &'static str {
        self.io.name()
    }

    fn register(m: &mut Meta, seq: u64, hdr: ManifestHeader, bytes: Arc<Vec<u8>>) {
        m.by_name.insert(hdr.name.clone(), seq);
        m.by_stream
            .entry(hdr.stream.clone())
            .or_default()
            .insert(hdr.mseq, seq);
        m.mans.insert(
            seq,
            ManEntry {
                wal_seq: seq,
                hdr,
                bytes,
            },
        );
    }

    fn lookup(&self, id: &ChunkId) -> Result<Option<Loc>> {
        Ok(self.db.get(id)?.and_then(|v| Loc::dec(&v)))
    }

    fn new_segment(&self) -> Result<(u32, Arc<File>)> {
        let id = self.next_seg.fetch_add(1, Relaxed);
        let f = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(seg_path(&self.cfg.root, id))?,
        );
        sync_dir(&self.cfg.root.join("segments"))?;
        self.segs.lock().unwrap().insert(id, f.clone());
        Ok((id, f))
    }

    fn rotate(&self, w: &mut Writer) -> Result<()> {
        let (id, f) = self.new_segment()?;
        *w = Writer {
            id,
            file: f,
            off: 0,
        };
        Ok(())
    }

    /// Append records to the active segment(s) and fdatasync. Returns the location of each record.
    fn flush(&self, batch: &[(ChunkId, Arc<Vec<u8>>)]) -> Result<Vec<(ChunkId, Loc)>> {
        let mut w = self.writer.lock().unwrap();
        let mut locs = Vec::with_capacity(batch.len());
        let mut buf: Vec<u8> = Vec::new();
        let mut start = w.off;
        for (id, data) in batch {
            let rl = seg_record_len(data.len()) as u64;
            if w.off > 0 && w.off + rl > self.cfg.seg_max {
                if !buf.is_empty() {
                    self.io.write_sync(&w.file, start, &buf)?;
                    self.stats.data_fsyncs.fetch_add(1, Relaxed);
                    buf.clear();
                }
                self.rotate(&mut w)?;
                start = 0;
            }
            locs.push((
                *id,
                Loc {
                    seg: w.id,
                    off: w.off,
                    len: data.len() as u32,
                    epoch: 0,
                },
            ));
            encode_seg_record(id, data, &mut buf);
            w.off += rl;
        }
        if !buf.is_empty() {
            self.io.write_sync(&w.file, start, &buf)?;
            self.stats.data_fsyncs.fetch_add(1, Relaxed);
        }
        let bytes: u64 = batch
            .iter()
            .map(|(_, d)| seg_record_len(d.len()) as u64)
            .sum();
        self.stats.seg_bytes_written.fetch_add(bytes, Relaxed);
        Ok(locs)
    }

    /// PUT_CHUNK for a batch. Returns only once every chunk is durable and indexed.
    pub fn put_batch(&self, chunks: Vec<(ChunkId, Vec<u8>)>) -> Result<PutReport> {
        let mut rep = PutReport::default();
        let mut mine: Vec<ChunkId> = Vec::new();
        let mut g = self.gq.lock().unwrap();
        if let Some(e) = &g.failed {
            bail!("store failed: {e}");
        }
        let ep = self.epoch.load(Relaxed);
        let my_ticket = g.next_ticket;
        let mut need = 0u64;
        let mut refresh = WriteBatch::default();
        let mut seen: HashSet<ChunkId> = HashSet::new();
        for (id, data) in chunks {
            self.stats
                .logical_bytes
                .fetch_add(data.len() as u64, Relaxed);
            if !seen.insert(id) {
                rep.dedup += 1;
                self.stats.dedup_hits.fetch_add(1, Relaxed);
                self.stats.dedup_bytes.fetch_add(data.len() as u64, Relaxed);
                continue;
            }
            if let Some(mut l) = self.lookup(&id)? {
                if l.epoch < ep && !cfg!(feature = "bug-no-epoch-refresh") {
                    l.epoch = ep;
                    refresh.put(id, l.enc());
                }
                rep.dedup += 1;
                self.stats.dedup_hits.fetch_add(1, Relaxed);
                self.stats.dedup_bytes.fetch_add(data.len() as u64, Relaxed);
            } else if let Some(&t) = g.pending_ids.get(&id) {
                need = need.max(t);
                rep.dedup += 1;
                self.stats.dedup_hits.fetch_add(1, Relaxed);
                self.stats.dedup_bytes.fetch_add(data.len() as u64, Relaxed);
            } else {
                g.pending_ids.insert(id, my_ticket);
                g.pending.push((id, Arc::new(data)));
                need = need.max(my_ticket);
                mine.push(id);
                rep.written += 1;
            }
        }
        if !mine.is_empty() {
            g.next_ticket += 1;
        }
        if !refresh.is_empty() {
            self.db.write_opt(refresh, &wopts(false))?;
        }
        loop {
            if let Some(e) = &g.failed {
                bail!("store failed: {e}");
            }
            if g.done_ticket >= need {
                break;
            }
            if g.flushing {
                g = self.cv.wait(g).unwrap();
                continue;
            }
            // become the group-commit leader
            let batch = std::mem::take(&mut g.pending);
            let upto = g.next_ticket - 1;
            g.flushing = true;
            drop(g);
            let res = self.flush(&batch);
            g = self.gq.lock().unwrap();
            let res = res.and_then(|locs| {
                // index entries only after the fsync above
                let ep = self.epoch.load(Relaxed);
                let mut wb = WriteBatch::default();
                for (id, mut l) in locs {
                    l.epoch = ep;
                    wb.put(id, l.enc());
                }
                self.db.write_opt(wb, &wopts(true))?;
                Ok(())
            });
            match res {
                Ok(()) => {
                    self.stats
                        .chunks_written
                        .fetch_add(batch.len() as u64, Relaxed);
                    for (id, _) in &batch {
                        g.pending_ids.remove(id);
                    }
                    g.done_ticket = upto;
                }
                Err(e) => g.failed = Some(format!("{e:#}")),
            }
            g.flushing = false;
            self.cv.notify_all();
        }
        drop(g);
        for id in mine {
            if let Some(l) = self.lookup(&id)? {
                rep.durable.push((l.seg, l.rec_end()));
            }
        }
        Ok(rep)
    }

    pub fn has(&self, ids: &[ChunkId]) -> Result<Vec<bool>> {
        ids.iter()
            .map(|id| Ok(self.lookup(id)?.is_some()))
            .collect()
    }

    /// COMMIT(manifest). Ack (Ok) only after the WAL record is fsync'd.
    pub fn commit(&self, bytes: Vec<u8>) -> std::result::Result<CommitReport, CommitError> {
        let hdr = parse_manifest(&bytes)?;
        let mut m = self.meta.lock().unwrap();
        if let Some(e) = &m.wal_error {
            return Err(anyhow!("store WAL corrupted: {e}").into());
        }
        if let Some((&last, _)) = m
            .by_stream
            .get(&hdr.stream)
            .and_then(|s| s.iter().next_back())
        {
            if hdr.mseq <= last {
                return Err(anyhow!(
                    "stale manifest seq {} for stream {:?} (last committed {last}): replay/rollback rejected",
                    hdr.mseq,
                    hdr.stream
                )
                .into());
            }
        }
        let ep = self.epoch.load(Relaxed);
        let mut missing = Vec::new();
        let mut refresh = WriteBatch::default();
        let mut seen = HashSet::new();
        for id in &hdr.ids {
            if !seen.insert(*id) {
                continue;
            }
            match self.lookup(id)? {
                Some(mut l) => {
                    if l.epoch < ep && !cfg!(feature = "bug-no-epoch-refresh") {
                        l.epoch = ep;
                        refresh.put(id, l.enc());
                    }
                }
                None => missing.push(*id),
            }
        }
        if !missing.is_empty() {
            return Err(CommitError::Missing(missing));
        }
        if !refresh.is_empty() {
            self.db
                .write_opt(refresh, &wopts(false))
                .map_err(anyhow::Error::from)?;
        }
        let seq = m.next_seq;
        let rec = encode_wal_record(seq, &bytes);
        if let Err(e) = self.io.write_sync(&m.wal, m.wal_end, &rec) {
            m.wal_error = Some(format!("WAL append failed: {e}"));
            return Err(anyhow!("WAL append failed: {e}").into());
        }
        m.wal_end += rec.len() as u64;
        m.next_seq += 1;
        self.stats
            .wal_bytes_written
            .fetch_add(rec.len() as u64, Relaxed);
        self.stats.commits.fetch_add(1, Relaxed);
        Self::register(&mut m, seq, hdr, Arc::new(bytes));
        Ok(CommitReport {
            wal_seq: seq,
            wal_end: m.wal_end,
        })
    }

    fn meta_ok(&self) -> Result<std::sync::MutexGuard<'_, Meta>> {
        let m = self.meta.lock().unwrap();
        if let Some(e) = &m.wal_error {
            bail!("store WAL corrupted: {e}");
        }
        Ok(m)
    }

    pub fn get_manifest(&self, name: &str) -> Result<Option<Arc<Vec<u8>>>> {
        let m = self.meta_ok()?;
        Ok(m.by_name
            .get(name)
            .and_then(|s| m.mans.get(s))
            .map(|e| e.bytes.clone()))
    }

    pub fn get_manifest_seq(&self, stream: &str, mseq: u64) -> Result<Option<Arc<Vec<u8>>>> {
        let m = self.meta_ok()?;
        Ok(m.by_stream
            .get(stream)
            .and_then(|s| s.get(&mseq))
            .and_then(|w| m.mans.get(w))
            .map(|e| e.bytes.clone()))
    }

    /// Latest committed manifest: of a stream, or (empty stream) the last accepted WAL record.
    pub fn latest(&self, stream: &str) -> Result<Option<Arc<Vec<u8>>>> {
        let m = self.meta_ok()?;
        if stream.is_empty() {
            return Ok(m.mans.values().next_back().map(|e| e.bytes.clone()));
        }
        Ok(m.by_stream
            .get(stream)
            .and_then(|s| s.values().next_back())
            .and_then(|w| m.mans.get(w))
            .map(|e| e.bytes.clone()))
    }

    pub fn list(&self) -> Result<Vec<serde_json::Value>> {
        let m = self.meta_ok()?;
        Ok(m.mans
            .values()
            .map(|e| {
                serde_json::json!({"wal_seq": e.wal_seq, "name": e.hdr.name, "stream": e.hdr.stream,
                    "mseq": e.hdr.mseq, "chunks": e.hdr.ids.len()})
            })
            .collect())
    }

    /// RESTORE path: read and re-verify (header CRC, chunk id, BLAKE3) every requested chunk.
    pub fn get_chunks(&self, ids: &[ChunkId]) -> Result<Vec<Vec<u8>>> {
        let mut reqs = Vec::with_capacity(ids.len());
        let mut lens = Vec::with_capacity(ids.len());
        for id in ids {
            // Look the location up *before* resolving the segment: a concurrent GC relocates the entry
            // before it unlinks the victim, so a vanished segment means "look again".
            let mut found = None;
            for _ in 0..3 {
                let l = self
                    .lookup(id)?
                    .ok_or_else(|| anyhow!("missing chunk {}", hex(id)))?;
                if let Some(f) = self.segs.lock().unwrap().get(&l.seg) {
                    found = Some((l, f.clone()));
                    break;
                }
            }
            let (l, f) =
                found.ok_or_else(|| anyhow!("chunk {} points at a missing segment", hex(id)))?;
            reqs.push(ReadReq {
                file: f,
                off: l.off,
                buf: vec![0u8; seg_record_len(l.len as usize)],
            });
            lens.push(l.len);
        }
        self.io
            .read_many(&mut reqs)
            .map_err(|e| anyhow!("chunk read failed: {e}"))?;
        let mut out = Vec::with_capacity(ids.len());
        for ((r, id), len) in reqs.into_iter().zip(ids).zip(lens) {
            let p =
                verify_seg_record(&r.buf, id, len).with_context(|| format!("chunk {}", hex(id)))?;
            out.push(p.to_vec());
        }
        Ok(out)
    }

    /// Scrub: verify every chunk of every retained manifest.
    pub fn verify(&self) -> Result<serde_json::Value> {
        let mans: Vec<ManEntry> = self.meta_ok()?.mans.values().cloned().collect();
        let mut bad = Vec::new();
        let mut ok_chunks = 0u64;
        for m in &mans {
            for id in &m.hdr.ids {
                match self.get_chunks(&[*id]) {
                    Ok(_) => ok_chunks += 1,
                    Err(e) => bad.push(
                        serde_json::json!({"manifest": m.hdr.name, "error": format!("{e:#}")}),
                    ),
                }
            }
        }
        Ok(serde_json::json!({"manifests": mans.len(), "chunks_ok": ok_chunks, "errors": bad}))
    }

    fn retained(m: &Meta, keep: usize) -> HashSet<u64> {
        let mut keepset = HashSet::new();
        for s in m.by_stream.values() {
            for w in s.values().rev().take(keep) {
                keepset.insert(*w);
            }
        }
        keepset
    }

    /// Rewrite the WAL with only the retained records (original seq numbers), atomically.
    fn compact_wal(&self, m: &mut Meta, keep: &HashSet<u64>) -> Result<usize> {
        let drop: Vec<u64> = m
            .mans
            .keys()
            .filter(|s| !keep.contains(s))
            .copied()
            .collect();
        if drop.is_empty() {
            return Ok(0);
        }
        let dir = self.cfg.root.join("wal");
        let tmp = dir.join("manifest.wal.tmp");
        let mut buf = Vec::new();
        for (seq, e) in &m.mans {
            if keep.contains(seq) {
                buf.extend_from_slice(&encode_wal_record(*seq, &e.bytes));
            }
        }
        {
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all_at(&buf, 0)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, dir.join("manifest.wal"))?;
        sync_dir(&dir)?;
        m.wal = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join("manifest.wal"))?;
        m.wal_end = buf.len() as u64;
        for s in &drop {
            if let Some(e) = m.mans.remove(s) {
                if let Some(st) = m.by_stream.get_mut(&e.hdr.stream) {
                    st.remove(&e.hdr.mseq);
                }
                if m.by_name.get(&e.hdr.name) == Some(s) {
                    m.by_name.remove(&e.hdr.name);
                }
            }
        }
        self.stats
            .wal_bytes_written
            .fetch_add(buf.len() as u64, Relaxed);
        Ok(drop.len())
    }

    fn hook(&self, phase: &str) {
        if let Some(h) = self.gc_hook.lock().unwrap().as_ref() {
            h(phase);
        }
    }

    /// Copy the records at `items` from `victim` into the GC output segment; returns new locations.
    fn gc_copy(
        &self,
        out: &mut Option<Writer>,
        victim: &Arc<File>,
        items: &[(ChunkId, Loc)],
    ) -> Result<Vec<(ChunkId, Loc, Loc)>> {
        let mut reqs: Vec<ReadReq> = items
            .iter()
            .map(|(_, l)| ReadReq {
                file: victim.clone(),
                off: l.off,
                buf: vec![0u8; seg_record_len(l.len as usize)],
            })
            .collect();
        self.io.read_many(&mut reqs)?;
        let mut res = Vec::new();
        let mut buf = Vec::new();
        for ((id, l), r) in items.iter().zip(&reqs) {
            verify_seg_record(&r.buf, id, l.len)
                .with_context(|| format!("GC copy of chunk {}", hex(id)))?;
            if out
                .as_ref()
                .is_none_or(|w| w.off > 0 && w.off + r.buf.len() as u64 > self.cfg.seg_max)
            {
                if let Some(w) = out.as_mut() {
                    if !buf.is_empty() {
                        self.io
                            .write_sync(&w.file, w.off - buf.len() as u64, &buf)?;
                        buf.clear();
                    }
                }
                let (sid, f) = self.new_segment()?;
                *out = Some(Writer {
                    id: sid,
                    file: f,
                    off: 0,
                });
            }
            let w = out.as_mut().unwrap();
            res.push((
                *id,
                *l,
                Loc {
                    seg: w.id,
                    off: w.off,
                    len: l.len,
                    epoch: l.epoch,
                },
            ));
            buf.extend_from_slice(&r.buf);
            w.off += r.buf.len() as u64;
        }
        if let Some(w) = out.as_ref() {
            if !buf.is_empty() {
                self.io
                    .write_sync(&w.file, w.off - buf.len() as u64, &buf)?;
            }
        }
        let n: u64 = reqs.iter().map(|r| r.buf.len() as u64).sum();
        self.stats.gc_bytes_copied.fetch_add(n, Relaxed);
        self.stats.seg_bytes_written.fetch_add(n, Relaxed);
        Ok(res)
    }

    /// Mark/copy/sweep GC retaining the last `keep` committed manifests of every stream.
    /// Sealed segments whose dead fraction is >= `min_dead` are compacted and deleted.
    pub fn gc(&self, keep: usize, min_dead: f64) -> Result<GcReport> {
        let t0 = Instant::now();
        let _one = self.gc_lock.lock().unwrap();
        let mut rep = GcReport::default();
        // ---- MARK (under meta + gq): new epoch, live set, WAL compaction, seal active segment.
        let (start, live, sealed_below) = {
            let mut m = self.meta.lock().unwrap();
            if let Some(e) = &m.wal_error {
                bail!("store WAL corrupted: {e}");
            }
            let g = self.gq.lock().unwrap();
            let start = self.epoch.fetch_add(1, Relaxed) + 1;
            let keepset = Self::retained(&m, keep);
            let mut live = HashSet::new();
            for s in &keepset {
                live.extend(m.mans[s].hdr.ids.iter().copied());
            }
            rep.retained_manifests = keepset.len();
            rep.dropped_manifests = self.compact_wal(&mut m, &keepset)?;
            let mut w = self.writer.lock().unwrap();
            self.rotate(&mut w)?;
            let below = w.id;
            drop(w);
            drop(g);
            (start, live, below)
        };
        rep.start_epoch = start;
        rep.live_chunks = live.len();
        self.hook("marked");

        // ---- choose victims (no locks; estimates only, correctness does not depend on them)
        let segs = self.segs.lock().unwrap().clone();
        let mut per_seg: HashMap<u32, Vec<(ChunkId, Loc)>> = HashMap::new();
        let mut live_bytes: HashMap<u32, u64> = HashMap::new();
        for kv in self.db.iterator(IteratorMode::Start) {
            let (k, v) = kv?;
            let (Some(l), Ok(id)) = (Loc::dec(&v), <ChunkId>::try_from(&k[..])) else {
                continue;
            };
            if l.seg >= sealed_below {
                continue;
            }
            if live.contains(&id) || l.epoch >= start {
                *live_bytes.entry(l.seg).or_default() += seg_record_len(l.len as usize) as u64;
            }
            per_seg.entry(l.seg).or_default().push((id, l));
        }
        let mut victims = Vec::new();
        for (&sid, f) in segs.range(..sealed_below) {
            let len = f.metadata()?.len();
            let lb = live_bytes.get(&sid).copied().unwrap_or(0);
            let dead = if len == 0 {
                1.0
            } else {
                1.0 - lb as f64 / len as f64
            };
            if dead >= min_dead || len == 0 {
                victims.push(sid);
            }
        }
        rep.victims = victims.clone();

        // ---- COPY live (and young) chunks out of victims, fsync, then relocate index entries.
        let mut out: Option<Writer> = None;
        for &sid in &victims {
            let f = &segs[&sid];
            let items: Vec<(ChunkId, Loc)> = per_seg
                .get(&sid)
                .map(|v| {
                    v.iter()
                        .filter(|(id, l)| live.contains(id) || l.epoch >= start)
                        .copied()
                        .collect()
                })
                .unwrap_or_default();
            let moved = self.gc_copy(&mut out, f, &items)?;
            let _m = self.meta.lock().unwrap();
            let _g = self.gq.lock().unwrap();
            let mut wb = WriteBatch::default();
            for (id, old, mut new) in moved {
                if let Some(cur) = self.lookup(&id)? {
                    if cur.seg == old.seg && cur.off == old.off {
                        new.epoch = cur.epoch;
                        wb.put(id, new.enc());
                        rep.copied_chunks += 1;
                        rep.copied_bytes += seg_record_len(new.len as usize) as u64;
                    }
                }
            }
            self.db.write_opt(wb, &wopts(true))?;
        }
        self.hook("copied");

        // ---- SWEEP: re-check every entry still in a victim immediately before deleting it.
        for &sid in &victims {
            let f = &segs[&sid];
            let m = self.meta.lock().unwrap();
            let g = self.gq.lock().unwrap();
            let mut keep_now = Vec::new();
            let mut wb = WriteBatch::default();
            for (id, _) in per_seg.get(&sid).map(|v| &v[..]).unwrap_or(&[]) {
                let Some(cur) = self.lookup(id)? else {
                    continue;
                };
                if cur.seg != sid {
                    continue;
                }
                if cur.epoch >= start || live.contains(id) {
                    keep_now.push((*id, cur));
                } else {
                    wb.delete(id);
                    rep.dropped_chunks += 1;
                }
            }
            if !keep_now.is_empty() {
                for (id, old, mut new) in self.gc_copy(&mut out, f, &keep_now)? {
                    new.epoch = old.epoch;
                    wb.put(id, new.enc());
                    rep.recheck_copies += 1;
                }
            }
            self.db.write_opt(wb, &wopts(true))?;
            rep.reclaimed_bytes += f.metadata()?.len();
            self.segs.lock().unwrap().remove(&sid);
            fs::remove_file(seg_path(&self.cfg.root, sid))?;
            sync_dir(&self.cfg.root.join("segments"))?;
            rep.deleted_segments += 1;
            drop(g);
            drop(m);
        }
        if let Some(w) = out {
            rep.reclaimed_bytes = rep.reclaimed_bytes.saturating_sub(w.off);
        }
        self.stats.gc_runs.fetch_add(1, Relaxed);
        self.stats
            .gc_segments_deleted
            .fetch_add(rep.deleted_segments, Relaxed);
        self.stats
            .gc_chunks_dropped
            .fetch_add(rep.dropped_chunks, Relaxed);
        rep.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(rep)
    }

    pub fn stats_json(&self) -> serde_json::Value {
        let s = &self.stats;
        let g = |a: &AtomicU64| a.load(Relaxed);
        let disk: u64 = self
            .segs
            .lock()
            .unwrap()
            .values()
            .filter_map(|f| f.metadata().ok())
            .map(|m| m.len())
            .sum();
        serde_json::json!({
            "io": self.io.name(),
            "logical_bytes": g(&s.logical_bytes), "dedup_hits": g(&s.dedup_hits), "dedup_bytes": g(&s.dedup_bytes),
            "chunks_written": g(&s.chunks_written), "seg_bytes_written": g(&s.seg_bytes_written),
            "wal_bytes_written": g(&s.wal_bytes_written), "data_fsyncs": g(&s.data_fsyncs),
            "commits": g(&s.commits), "gc_runs": g(&s.gc_runs), "gc_bytes_copied": g(&s.gc_bytes_copied),
            "gc_segments_deleted": g(&s.gc_segments_deleted), "gc_chunks_dropped": g(&s.gc_chunks_dropped),
            "segment_disk_bytes": disk, "epoch": self.epoch.load(Relaxed),
            "index_disk_bytes": dir_size(&self.cfg.root.join("index")),
            "wal_disk_bytes": dir_size(&self.cfg.root.join("wal")),
        })
    }

    /// Ensure a given payload is what the record claims (used by tests).
    pub fn chunk_loc(&self, id: &ChunkId) -> Result<Option<Loc>> {
        self.lookup(id)
    }

    pub fn check_invariant_i1(&self) -> Result<()> {
        let m = self.meta_ok()?;
        for e in m.mans.values() {
            for id in &e.hdr.ids {
                ensure!(
                    self.lookup(id)?.is_some(),
                    "I1 violated: {} references unindexed chunk",
                    e.hdr.name
                );
            }
        }
        Ok(())
    }
}

fn dir_size(p: &Path) -> u64 {
    fs::read_dir(p)
        .map(|rd| {
            rd.filter_map(|e| e.ok()?.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}
