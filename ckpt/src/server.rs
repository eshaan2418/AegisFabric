//! Length-prefixed binary TCP protocol.
//!
//! Request:  len(u32 LE, = 1 + payload) | op(u8) | payload
//! Response: len(u32 LE, = 1 + payload) | status(u8) | payload
//! status: 0 OK, 1 ERROR, 2 CORRUPT (detected integrity violation), 3 NOT_FOUND, 4 INVALID
//!
//! Mutations (PUT_CHUNK(S), COMMIT, GC) are funnelled through a single writer thread that batches
//! concurrent PUTs from all connections into one durable append + fsync (group commit).

use crate::format::Id;
use crate::store::{Error, Result, Store};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

pub const OP_PUT_CHUNK: u8 = 1;
pub const OP_HAS_CHUNKS: u8 = 2;
pub const OP_COMMIT: u8 = 3;
pub const OP_LATEST: u8 = 4;
pub const OP_GET_MANIFEST: u8 = 5;
pub const OP_GET_CHUNK: u8 = 6;
pub const OP_GC: u8 = 7;
pub const OP_STATS: u8 = 8;
pub const OP_PUT_CHUNKS: u8 = 9;
pub const OP_VERIFY: u8 = 10;
pub const OP_LIST: u8 = 11;

const GROUP_MAX_BYTES: usize = 128 << 20;
const MAX_FRAME: usize = 512 << 20;

enum Mutation {
    Put(Vec<(Id, Vec<u8>)>, Sender<Result<Vec<bool>>>),
    Commit(u64, Vec<Id>, Vec<u8>, Sender<Result<()>>),
    Gc(usize, Sender<Result<serde_json::Value>>),
}

fn writer_loop(store: Arc<Mutex<Store>>, rx: Receiver<Mutation>) {
    let mut carry: Option<Mutation> = None;
    loop {
        let first = match carry.take() {
            Some(m) => m,
            None => match rx.recv() {
                Ok(m) => m,
                Err(_) => return,
            },
        };
        match first {
            Mutation::Put(chunks, tx) => {
                let mut group = vec![(chunks, tx)];
                let mut bytes: usize = group[0].0.iter().map(|c| c.1.len()).sum();
                while bytes < GROUP_MAX_BYTES {
                    match rx.try_recv() {
                        Ok(Mutation::Put(c, t)) => {
                            bytes += c.iter().map(|x| x.1.len()).sum::<usize>();
                            group.push((c, t));
                        }
                        Ok(other) => {
                            carry = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                let flat: Vec<(Id, &[u8])> = group
                    .iter()
                    .flat_map(|(c, _)| c.iter().map(|(i, p)| (*i, p.as_slice())))
                    .collect();
                let res = store.lock().unwrap().put_chunks(&flat);
                match res {
                    Ok(flags) => {
                        let mut k = 0;
                        for (c, t) in group {
                            let n = c.len();
                            let _ = t.send(Ok(flags[k..k + n].to_vec()));
                            k += n;
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        for (_, t) in group {
                            let _ = t.send(Err(Error::Io(std::io::Error::other(msg.clone()))));
                        }
                    }
                }
            }
            Mutation::Commit(seq, ids, blob, tx) => {
                let _ = tx.send(store.lock().unwrap().commit(seq, &ids, &blob));
            }
            Mutation::Gc(keep, tx) => {
                let _ = tx.send(store.lock().unwrap().gc(keep));
            }
        }
    }
}

fn status_of(e: &Error) -> u8 {
    match e {
        Error::Io(_) => 1,
        Error::Corrupt(_) => 2,
        Error::NotFound(_) => 3,
        Error::Invalid(_) => 4,
    }
}

fn read_frame(r: &mut impl Read) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut lb = [0u8; 4];
    match r.read_exact(&mut lb) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(lb) as usize;
    if len == 0 || len > MAX_FRAME {
        return Err(std::io::Error::other("bad frame length"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    let op = buf[0];
    buf.remove(0);
    Ok(Some((op, buf)))
}

fn write_resp(w: &mut impl Write, status: u8, payload: &[u8]) -> std::io::Result<()> {
    w.write_all(&((payload.len() + 1) as u32).to_le_bytes())?;
    w.write_all(&[status])?;
    w.write_all(payload)?;
    w.flush()
}

fn id_at(p: &[u8], at: usize) -> Result<Id> {
    p.get(at..at + 32)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| Error::Invalid("truncated id".into()))
}
fn u32_at(p: &[u8], at: usize) -> Result<u32> {
    p.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::Invalid("truncated u32".into()))
}
fn u64_at(p: &[u8], at: usize) -> Result<u64> {
    p.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::Invalid("truncated u64".into()))
}

fn ids_at(p: &[u8], at: usize, n: usize) -> Result<Vec<Id>> {
    (0..n).map(|i| id_at(p, at + 32 * i)).collect()
}

fn handle(op: u8, p: Vec<u8>, store: &Arc<Mutex<Store>>, tx: &Sender<Mutation>) -> Result<Vec<u8>> {
    match op {
        OP_PUT_CHUNK => {
            let id = id_at(&p, 0)?;
            let (rt, rr) = channel();
            tx.send(Mutation::Put(vec![(id, p[32..].to_vec())], rt))
                .map_err(|_| Error::Invalid("writer gone".into()))?;
            let f = rr
                .recv()
                .map_err(|_| Error::Invalid("writer gone".into()))??;
            Ok(vec![f[0] as u8])
        }
        OP_PUT_CHUNKS => {
            let n = u32_at(&p, 0)? as usize;
            let mut at = 4;
            let mut chunks = Vec::with_capacity(n);
            for _ in 0..n {
                let id = id_at(&p, at)?;
                let len = u32_at(&p, at + 32)? as usize;
                let s = at + 36;
                let body = p
                    .get(s..s + len)
                    .ok_or_else(|| Error::Invalid("truncated chunk".into()))?;
                chunks.push((id, body.to_vec()));
                at = s + len;
            }
            let (rt, rr) = channel();
            tx.send(Mutation::Put(chunks, rt))
                .map_err(|_| Error::Invalid("writer gone".into()))?;
            let f = rr
                .recv()
                .map_err(|_| Error::Invalid("writer gone".into()))??;
            Ok(f.into_iter().map(|b| b as u8).collect())
        }
        OP_HAS_CHUNKS => {
            let n = u32_at(&p, 0)? as usize;
            let ids = ids_at(&p, 4, n)?;
            let r = store.lock().unwrap().has_chunks(&ids)?;
            Ok(r.into_iter().map(|b| b as u8).collect())
        }
        OP_COMMIT => {
            let seq = u64_at(&p, 0)?;
            let n = u32_at(&p, 8)? as usize;
            let ids = ids_at(&p, 12, n)?;
            let blob = p[12 + 32 * n..].to_vec();
            let (rt, rr) = channel();
            tx.send(Mutation::Commit(seq, ids, blob, rt))
                .map_err(|_| Error::Invalid("writer gone".into()))?;
            rr.recv()
                .map_err(|_| Error::Invalid("writer gone".into()))??;
            Ok(vec![])
        }
        OP_LATEST => match store.lock().unwrap().latest()? {
            Some(s) => Ok(s.to_le_bytes().to_vec()),
            None => Err(Error::NotFound("no committed checkpoint".into())),
        },
        OP_LIST => {
            let l = store.lock().unwrap().list();
            let mut out = (l.len() as u32).to_le_bytes().to_vec();
            for s in l {
                out.extend_from_slice(&s.to_le_bytes());
            }
            Ok(out)
        }
        OP_GET_MANIFEST => {
            let seq = u64_at(&p, 0)?;
            let (s, ids, blob) = store.lock().unwrap().get_manifest(seq)?;
            let mut out = s.to_le_bytes().to_vec();
            out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
            for i in ids {
                out.extend_from_slice(&i);
            }
            out.extend_from_slice(&blob);
            Ok(out)
        }
        OP_GET_CHUNK => {
            let id = id_at(&p, 0)?;
            store.lock().unwrap().get_chunk(&id)
        }
        OP_GC => {
            let keep = u32_at(&p, 0)? as usize;
            let (rt, rr) = channel();
            tx.send(Mutation::Gc(keep, rt))
                .map_err(|_| Error::Invalid("writer gone".into()))?;
            let v = rr
                .recv()
                .map_err(|_| Error::Invalid("writer gone".into()))??;
            Ok(v.to_string().into_bytes())
        }
        OP_STATS => Ok(store.lock().unwrap().stats().to_string().into_bytes()),
        OP_VERIFY => {
            let seq = u64_at(&p, 0)?;
            Ok(store.lock().unwrap().verify(seq)?.to_string().into_bytes())
        }
        _ => Err(Error::Invalid(format!("unknown op {op}"))),
    }
}

fn serve_conn(s: TcpStream, store: Arc<Mutex<Store>>, tx: Sender<Mutation>) {
    let _ = s.set_nodelay(true);
    let mut r = BufReader::with_capacity(1 << 20, s.try_clone().expect("clone"));
    let mut w = BufWriter::with_capacity(1 << 20, s);
    loop {
        let (op, p) = match read_frame(&mut r) {
            Ok(Some(x)) => x,
            _ => return,
        };
        let res = handle(op, p, &store, &tx);
        let ok = match res {
            Ok(out) => write_resp(&mut w, 0, &out),
            Err(e) => write_resp(&mut w, status_of(&e), e.to_string().as_bytes()),
        };
        if ok.is_err() {
            return;
        }
    }
}

pub fn serve(store: Store, listen: &str) -> std::io::Result<()> {
    let l = TcpListener::bind(listen)?;
    let store = Arc::new(Mutex::new(store));
    let (tx, rx) = channel();
    {
        let st = store.clone();
        std::thread::spawn(move || writer_loop(st, rx));
    }
    {
        let st = store.lock().unwrap();
        println!("RECOVERY {}", st.recovery);
    }
    println!("READY {}", l.local_addr()?);
    std::io::stdout().flush()?;
    for c in l.incoming() {
        let c = match c {
            Ok(c) => c,
            Err(_) => continue,
        };
        let st = store.clone();
        let t = tx.clone();
        std::thread::spawn(move || serve_conn(c, st, t));
    }
    Ok(())
}
