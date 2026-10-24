use aegis_ckpt::io::IoKind;
use aegis_ckpt::store::Store;
use std::path::PathBuf;
use std::process::exit;

const USAGE: &str = "usage:
  aegis-ckpt serve   --root DIR --listen HOST:PORT [--io uring|pwrite] [--io-threads N]
  aegis-ckpt recover --root DIR [--io uring|pwrite]      # run recovery, print report, exit
  aegis-ckpt verify  --root DIR [--seq N]                 # verify a manifest's chunks (0 = latest)";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{USAGE}");
        exit(2);
    }
    let mut root: Option<PathBuf> = None;
    let mut listen = "127.0.0.1:7070".to_string();
    let mut io = IoKind::Pwrite;
    let mut threads = 4usize;
    let mut seq = 0u64;
    let mut i = 2;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--root" => root = Some(PathBuf::from(v)),
            "--listen" => listen = v,
            "--io" => {
                io = IoKind::parse(&v).unwrap_or_else(|| {
                    eprintln!("bad --io {v}");
                    exit(2)
                })
            }
            "--io-threads" => threads = v.parse().unwrap_or(4),
            "--seq" => seq = v.parse().unwrap_or(0),
            _ => {
                eprintln!("unknown arg {}\n{USAGE}", args[i]);
                exit(2)
            }
        }
        i += 2;
    }
    let Some(root) = root else {
        eprintln!("{USAGE}");
        exit(2)
    };
    let store = match Store::open(&root, io, threads) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("open failed: {e}");
            exit(1)
        }
    };
    match args[1].as_str() {
        "serve" => {
            if let Err(e) = aegis_ckpt::server::serve(store, &listen) {
                eprintln!("serve: {e}");
                exit(1)
            }
        }
        "recover" => println!("{}", store.recovery),
        "verify" => match store.verify(seq) {
            Ok(v) => {
                println!("{v}");
                if v["ok"] != true {
                    exit(3)
                }
            }
            Err(e) => {
                println!("{{\"error\": \"{e}\"}}");
                exit(3)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            exit(2)
        }
    }
}
