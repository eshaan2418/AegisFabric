//! Chunking: fixed-size and content-defined (FastCDC-style gear hash with normalized chunking,
//! Xia et al., USENIX ATC'16).
//!
//! Which one is used where (see results/m4/NOTES.md for the measurement behind the choice):
//! the network client chunks *before* encryption (P2 needs keyed chunk ids over plaintext), so
//! chunking happens client-side. torch.save writes each tensor as its own 64-byte-aligned zip
//! member, and tensor sizes never change between training steps, so byte offsets are stable across
//! successive checkpoints; fixed-size chunking then dedups as well as CDC (measured with
//! `aegis-ckpt chunkscan`). CDC is kept for the analysis and for inputs with shifting offsets.

/// Chunk boundaries (exclusive end offsets) for fixed-size chunking.
pub fn fixed(len: usize, size: usize) -> Vec<usize> {
    assert!(size > 0);
    let mut v: Vec<usize> = (1..=len / size).map(|i| i * size).collect();
    if !len.is_multiple_of(size) || len == 0 {
        v.push(len);
    }
    v
}

fn gear_table() -> [u64; 256] {
    // splitmix64, fixed seed: deterministic table shared by every build.
    let mut t = [0u64; 256];
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for e in t.iter_mut() {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *e = z ^ (z >> 31);
    }
    t
}

/// Content-defined chunk boundaries. `avg` must be a power of two; min = avg/4, max = avg*4.
pub fn cdc(data: &[u8], avg: usize) -> Vec<usize> {
    assert!(avg.is_power_of_two() && avg >= 256);
    let gear = gear_table();
    let bits = avg.trailing_zeros();
    // Normalized chunking: a stricter mask before the average size, a looser one after it.
    // Gear hashes mix into the high bits (h << 1 + g), so the masks select high bits.
    let mask_s: u64 = !0u64 << (64 - (bits + 1));
    let mask_l: u64 = !0u64 << (64 - (bits - 1));
    let (min, max) = (avg / 4, avg * 4);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < data.len() {
        let rem = data.len() - start;
        if rem <= min {
            out.push(data.len());
            break;
        }
        let end_max = start + rem.min(max);
        let normal = (start + avg).min(end_max);
        let mut h: u64 = 0;
        let mut i = start + min;
        let mut cut = end_max;
        while i < normal {
            h = (h << 1).wrapping_add(gear[data[i] as usize]);
            if h & mask_s == 0 {
                cut = i + 1;
                break;
            }
            i += 1;
        }
        if cut == end_max && i >= normal {
            while i < end_max {
                h = (h << 1).wrapping_add(gear[data[i] as usize]);
                if h & mask_l == 0 {
                    cut = i + 1;
                    break;
                }
                i += 1;
            }
        }
        out.push(cut);
        start = cut;
    }
    if data.is_empty() {
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn fixed_covers_input() {
        assert_eq!(fixed(10, 4), vec![4, 8, 10]);
        assert_eq!(fixed(8, 4), vec![4, 8]);
        assert_eq!(fixed(0, 4), vec![0]);
    }

    #[test]
    fn cdc_bounds_and_shift_resistance() {
        let mut r = rand::rngs::StdRng::seed_from_u64(1);
        let mut d = vec![0u8; 4 << 20];
        r.fill_bytes(&mut d);
        let b = cdc(&d, 16384);
        assert_eq!(*b.last().unwrap(), d.len());
        let mut prev = 0;
        for &e in &b {
            assert!(e - prev <= 65536);
            if e != d.len() {
                assert!(e - prev >= 4096);
            }
            prev = e;
        }
        let mean = d.len() as f64 / b.len() as f64;
        assert!(mean > 8000.0 && mean < 40000.0, "mean chunk {mean}");
        // insert 100 bytes near the start: most boundaries re-synchronise (shifted by 100)
        let mut d2 = d[..1000].to_vec();
        d2.extend_from_slice(&[42u8; 100]);
        d2.extend_from_slice(&d[1000..]);
        let b2: std::collections::HashSet<usize> = cdc(&d2, 16384).into_iter().collect();
        let same = b.iter().filter(|&&e| b2.contains(&(e + 100))).count();
        assert!(same as f64 > 0.9 * b.len() as f64, "{same}/{}", b.len());
    }
}
