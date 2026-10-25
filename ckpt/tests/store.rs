//! Store-level tests of protocol P1: commit rules, recovery, corruption detection, GC (incl. the
//! TLC-found interleaving where a COMMIT lands between GC mark and sweep).

use aegis_ckpt::format::{encode_manifest, ChunkId, ManifestHeader};
use aegis_ckpt::store::{CommitError, Config, Store};
use rand::{Rng, RngCore, SeedableRng};
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn cfg(p: &Path) -> Config {
    let mut c = Config::new(p);
    c.io = "auto".into();
    c.seg_max = 1 << 20;
    c
}

fn id_of(d: &[u8]) -> ChunkId {
    *blake3::hash(d).as_bytes()
}

fn chunks(seed: u64, n: usize, sz: usize) -> Vec<Vec<u8>> {
    let mut r = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v = vec![0u8; sz];
            r.fill_bytes(&mut v);
            v
        })
        .collect()
}

fn manifest(name: &str, stream: &str, mseq: u64, data: &[Vec<u8>]) -> Vec<u8> {
    let h = ManifestHeader {
        name: name.into(),
        stream: stream.into(),
        mseq,
        ids: data.iter().map(|d| id_of(d)).collect(),
    };
    encode_manifest(&h, b"body")
}

fn put_all(s: &Store, data: &[Vec<u8>]) {
    s.put_batch(data.iter().map(|d| (id_of(d), d.clone())).collect())
        .unwrap();
}

fn restore(s: &Store, data: &[Vec<u8>]) -> anyhow::Result<Vec<Vec<u8>>> {
    s.get_chunks(&data.iter().map(|d| id_of(d)).collect::<Vec<_>>())
}

#[test]
fn roundtrip_dedup_and_commit_rules() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::open(cfg(t.path())).unwrap();
    let a = chunks(1, 20, 10_000);
    put_all(&s, &a);
    let r = s
        .put_batch(a.iter().map(|d| (id_of(d), d.clone())).collect())
        .unwrap();
    assert_eq!((r.dedup, r.written), (20, 0));
    s.commit(manifest("k1", "s", 1, &a)).unwrap();
    assert_eq!(restore(&s, &a).unwrap(), a);
    // missing chunk -> rejected with the missing id
    let b = chunks(2, 2, 100);
    match s.commit(manifest("k2", "s", 2, &b)) {
        Err(CommitError::Missing(m)) => assert_eq!(m.len(), 2),
        other => panic!("expected Missing, got {other:?}"),
    }
    // replayed / rolled-back sequence number -> rejected
    assert!(s.commit(manifest("k1", "s", 1, &a)).is_err());
    put_all(&s, &b);
    s.commit(manifest("k2", "s", 2, &b)).unwrap();
    drop(s);
    let s = Store::open(cfg(t.path())).unwrap();
    assert_eq!(s.recovery.wal_records, 2);
    assert!(s.latest("s").unwrap().unwrap().as_slice() == manifest("k2", "s", 2, &b).as_slice());
    assert_eq!(restore(&s, &a).unwrap(), a);
    s.check_invariant_i1().unwrap();
}

#[test]
fn torn_wal_tail_is_truncated() {
    let t = tempfile::tempdir().unwrap();
    let a = chunks(3, 4, 5000);
    {
        let s = Store::open(cfg(t.path())).unwrap();
        put_all(&s, &a);
        s.commit(manifest("k1", "s", 1, &a)).unwrap();
        s.commit(manifest("k2", "s", 2, &a)).unwrap();
    }
    let w = t.path().join("wal/manifest.wal");
    let len = std::fs::metadata(&w).unwrap().len();
    let f = OpenOptions::new().write(true).open(&w).unwrap();
    f.set_len(len - 7).unwrap();
    let s = Store::open(cfg(t.path())).unwrap();
    assert_eq!(s.recovery.wal_records, 1);
    assert!(s.recovery.wal_error.is_none());
    assert!(s.recovery.wal_truncated_bytes > 0);
    assert_eq!(
        s.latest("").unwrap().unwrap().as_slice(),
        manifest("k1", "s", 1, &a).as_slice()
    );
    // the next commit continues after the truncated tail
    s.commit(manifest("k3", "s", 3, &a)).unwrap();
    drop(s);
    let s = Store::open(cfg(t.path())).unwrap();
    assert_eq!(s.recovery.wal_records, 2);
}

#[test]
fn corrupt_wal_middle_is_a_detected_error() {
    let t = tempfile::tempdir().unwrap();
    let a = chunks(4, 2, 1000);
    {
        let s = Store::open(cfg(t.path())).unwrap();
        put_all(&s, &a);
        for i in 1..=3 {
            s.commit(manifest(&format!("k{i}"), "s", i, &a)).unwrap();
        }
    }
    let w = t.path().join("wal/manifest.wal");
    let f = OpenOptions::new().read(true).write(true).open(&w).unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 20).unwrap();
    f.write_all_at(&[b[0] ^ 0x04], 20).unwrap();
    let s = Store::open(cfg(t.path())).unwrap();
    assert!(s.recovery.wal_error.is_some());
    assert!(s.latest("").is_err());
    assert!(s.get_manifest("k3").is_err());
}

#[test]
fn bitflip_in_chunk_is_detected_and_index_past_end_dropped() {
    let t = tempfile::tempdir().unwrap();
    let a = chunks(5, 8, 4096);
    {
        let s = Store::open(cfg(t.path())).unwrap();
        put_all(&s, &a);
        s.commit(manifest("k", "s", 1, &a)).unwrap();
    }
    let seg = t.path().join("segments/seg-00000001.log");
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&seg)
        .unwrap();
    // flip one payload bit of the first record
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, 100).unwrap();
    f.write_all_at(&[b[0] ^ 0x80], 100).unwrap();
    {
        let s = Store::open(cfg(t.path())).unwrap();
        let e = restore(&s, &a).unwrap_err();
        assert!(format!("{e:#}").contains("blake3"), "{e:#}");
        let v = s.verify().unwrap();
        assert_eq!(v["errors"].as_array().unwrap().len(), 1);
    }
    // truncate the segment in the middle of the last record -> its index entry is dropped
    let len = f.metadata().unwrap().len();
    f.set_len(len - 10).unwrap();
    let s = Store::open(cfg(t.path())).unwrap();
    assert_eq!(s.recovery.index_dropped, 1);
    assert_eq!(s.recovery.latest_missing_chunks, 1);
    assert!(s.check_invariant_i1().is_err());
    assert!(restore(&s, &a[7..]).is_err());
}

#[test]
fn corrupt_index_is_rebuilt_from_segments() {
    let t = tempfile::tempdir().unwrap();
    let a = chunks(6, 10, 3000);
    {
        let s = Store::open(cfg(t.path())).unwrap();
        put_all(&s, &a);
        s.commit(manifest("k", "s", 1, &a)).unwrap();
    }
    // clobber RocksDB's MANIFEST/CURRENT so it cannot open
    for e in std::fs::read_dir(t.path().join("index")).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap().to_string_lossy().to_string();
        if n == "CURRENT" || n.starts_with("MANIFEST") {
            std::fs::write(&p, b"garbage\n").unwrap();
        }
    }
    let s = Store::open(cfg(t.path())).unwrap();
    assert!(s.recovery.index_rebuilt, "{:?}", s.recovery);
    assert_eq!(restore(&s, &a).unwrap(), a);
}

#[test]
fn gc_retains_last_k_and_reclaims() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::open(cfg(t.path())).unwrap();
    let base = chunks(7, 30, 20_000); // shared across versions (dedup)
    let mut versions = Vec::new();
    for v in 0..5u64 {
        let mut d = base.clone();
        d.extend(chunks(100 + v, 30, 20_000)); // per-version delta
        put_all(&s, &d);
        s.commit(manifest(&format!("k{v}"), "s", v + 1, &d))
            .unwrap();
        versions.push(d);
    }
    let disk0: u64 = s.stats_json()["segment_disk_bytes"].as_u64().unwrap();
    let r = s.gc(2, 0.0).unwrap();
    assert_eq!(r.dropped_manifests, 3);
    assert!(r.deleted_segments > 0 && r.dropped_chunks == 90, "{r:?}");
    let disk1: u64 = s.stats_json()["segment_disk_bytes"].as_u64().unwrap();
    assert!(disk1 < disk0 * 3 / 4, "{disk0} -> {disk1}");
    for v in &versions[3..5] {
        assert_eq!(&restore(&s, v).unwrap(), v);
    }
    assert!(s.get_manifest("k0").unwrap().is_none());
    assert!(restore(&s, &versions[0][30..31]).is_err());
    drop(s);
    let s = Store::open(cfg(t.path())).unwrap();
    assert_eq!(s.recovery.wal_records, 2);
    assert_eq!(s.recovery.index_dropped, 0);
    s.check_invariant_i1().unwrap();
    for v in &versions[3..5] {
        assert_eq!(&restore(&s, v).unwrap(), v);
    }
}

/// The TLC counterexample for the as-written P1: a COMMIT that dedups against a chunk that is dead at
/// GC mark time, landing between mark and sweep. With the epoch refresh the chunk must survive.
#[test]
fn commit_between_gc_mark_and_sweep_keeps_deduped_chunks() {
    let t = tempfile::tempdir().unwrap();
    let s = Arc::new(Store::open(cfg(t.path())).unwrap());
    let old = chunks(8, 6, 8000);
    put_all(&s, &old);
    s.commit(manifest("old", "a", 1, &old)).unwrap();
    let other = chunks(9, 3, 8000);
    put_all(&s, &other);
    s.commit(manifest("new1", "a", 2, &other)).unwrap(); // with keep=1, "old" is now dead
    let s2 = s.clone();
    let old2 = old.clone();
    let fired = Arc::new(Mutex::new(false));
    let f2 = fired.clone();
    *s.gc_hook.lock().unwrap() = Some(Box::new(move |phase: &str| {
        if phase == "marked" {
            // stream b re-uses all of "old"'s chunks: dedup hits, then COMMIT
            put_all(&s2, &old2);
            s2.commit(manifest("revive", "b", 1, &old2)).unwrap();
            *f2.lock().unwrap() = true;
        }
    }));
    let r = s.gc(1, 0.0).unwrap();
    assert!(*fired.lock().unwrap());
    assert!(r.deleted_segments > 0, "{r:?}");
    assert_eq!(
        restore(&s, &old).unwrap(),
        old,
        "deduped chunks of a commit during GC were swept"
    );
    s.check_invariant_i1().unwrap();
}

#[test]
fn commit_between_copy_and_sweep_keeps_deduped_chunks() {
    let t = tempfile::tempdir().unwrap();
    let s = Arc::new(Store::open(cfg(t.path())).unwrap());
    let old = chunks(10, 6, 8000);
    put_all(&s, &old);
    s.commit(manifest("old", "a", 1, &old)).unwrap();
    let other = chunks(11, 3, 8000);
    put_all(&s, &other);
    s.commit(manifest("new1", "a", 2, &other)).unwrap();
    let s2 = s.clone();
    let old2 = old.clone();
    *s.gc_hook.lock().unwrap() = Some(Box::new(move |phase: &str| {
        if phase == "copied" {
            s2.commit(manifest("revive", "b", 1, &old2)).unwrap(); // pure dedup commit, no PUT
        }
    }));
    let r = s.gc(1, 0.0).unwrap();
    assert!(r.recheck_copies > 0, "{r:?}");
    assert_eq!(restore(&s, &old).unwrap(), old);
}

#[test]
fn concurrent_writers_group_commit() {
    let t = tempfile::tempdir().unwrap();
    let s = Arc::new(Store::open(cfg(t.path())).unwrap());
    let shared = chunks(12, 10, 4000);
    let hs: Vec<_> = (0..4u64)
        .map(|w| {
            let s = s.clone();
            let shared = shared.clone();
            std::thread::spawn(move || {
                let mut r = rand::rngs::StdRng::seed_from_u64(w);
                for v in 1..=10u64 {
                    let mut d = shared.clone();
                    d.extend(chunks(1000 * w + v, r.gen_range(1..5), 3000));
                    put_all(&s, &d);
                    s.commit(manifest(&format!("w{w}v{v}"), &format!("w{w}"), v, &d))
                        .unwrap();
                    assert_eq!(restore(&s, &d).unwrap(), d);
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let st = s.stats_json();
    assert_eq!(st["commits"], 40);
    assert!(st["data_fsyncs"].as_u64().unwrap() <= 40);
    s.check_invariant_i1().unwrap();
}
