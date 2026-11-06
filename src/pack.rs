//! Bayer plane -> `[4, H/2, W/2]` channel stack (R, Gr, Gb, B), for ML input.
//!
//! Pure safe Rust with no indexing: every access goes through `chunks_exact`,
//! `as_chunks` or zipped iterators, so a bad shape is an `Err`, not a panic.

/// A position inside the repeating 2x2 tile, relative to the plane origin
/// (the same convention `irradiance::demosaic` uses for `CFAPattern`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Slot {
    /// Row-major order, matching `CFAPattern`'s byte order.
    const ALL: [Slot; 4] = [Slot::TopLeft, Slot::TopRight, Slot::BottomLeft, Slot::BottomRight];

    /// Row parity inside the tile (0 = top).
    pub fn row(self) -> u32 {
        match self {
            Slot::TopLeft | Slot::TopRight => 0,
            Slot::BottomLeft | Slot::BottomRight => 1,
        }
    }

    /// Column parity inside the tile (0 = left).
    pub fn col(self) -> u32 {
        match self {
            Slot::TopLeft | Slot::BottomLeft => 0,
            Slot::TopRight | Slot::BottomRight => 1,
        }
    }

    fn pick(self, tl: u16, tr: u16, bl: u16, br: u16) -> u16 {
        match self {
            Slot::TopLeft => tl,
            Slot::TopRight => tr,
            Slot::BottomLeft => bl,
            Slot::BottomRight => br,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Colour {
    Red,
    Green,
    Blue,
}

/// Where each output channel sits in the tile: `[R, Gr, Gb, B]`, where Gr is
/// the green on red's row and Gb the green on blue's row.
pub struct Phase {
    pub slots: [Slot; 4],
}

impl Phase {
    /// Validate a 2x2 Bayer CFA (any of RGGB/GRBG/GBRG/BGGR) and locate its
    /// channels. Colours resolve through `CFAPlaneColor` the way upstream's
    /// demosaic does, falling back to the pattern index itself.
    pub fn from_cfa(repeat_dim: [u32; 2], pattern: &[u8], plane_color: &[u8]) -> Result<Phase, String> {
        let not_bayer = || format!("CFA {repeat_dim:?} {pattern:?} is not a 2x2 Bayer mosaic");
        if repeat_dim != [2, 2] || pattern.len() != 4 {
            return Err(not_bayer());
        }
        let mut colours = Vec::with_capacity(4);
        for (&idx, slot) in pattern.iter().zip(Slot::ALL) {
            let colour = match plane_color.get(usize::from(idx)).copied().unwrap_or(idx) {
                0 => Colour::Red,
                1 => Colour::Green,
                2 => Colour::Blue,
                _ => return Err(not_bayer()),
            };
            colours.push((slot, colour));
        }
        let find = |want: Colour| -> Vec<Slot> {
            colours.iter().filter(|(_, c)| *c == want).map(|(s, _)| *s).collect()
        };
        let (reds, greens, blues) = (find(Colour::Red), find(Colour::Green), find(Colour::Blue));
        let (&[r], &[b], 2) = (reds.as_slice(), blues.as_slice(), greens.len()) else {
            return Err(not_bayer());
        };
        // Red and blue diagonal <=> greens diagonal: the defining Bayer property.
        if r.row() == b.row() || r.col() == b.col() {
            return Err(not_bayer());
        }
        let same_row_green = |s: Slot| greens.iter().copied().find(|g| g.row() == s.row());
        let (Some(gr), Some(gb)) = (same_row_green(r), same_row_green(b)) else {
            return Err(not_bayer());
        };
        Ok(Phase { slots: [r, gr, gb, b] })
    }
}

/// Pack a row-major plane of width `w` into `dst` = 4 planes of `(h/2, w/2)`,
/// applying `conv[k]` to channel `k`. An odd trailing row or column is dropped.
pub fn pack<T, C: Fn(u16) -> T>(
    src: &[u16],
    w: usize,
    phase: &Phase,
    dst: &mut [T],
    conv: &[C; 4],
) -> Result<(), String> {
    let h = src.len().checked_div(w).unwrap_or(0);
    let (h2, w2) = (h / 2, w / 2);
    let n = h2 * w2;
    if n == 0 || src.len() != h * w {
        return Err(format!("plane of {} samples, width {w}, is too small to pack", src.len()));
    }
    if dst.len() != 4 * n {
        return Err(format!("out holds {} values, need 4 * {h2} * {w2}", dst.len()));
    }

    let mut planes = dst.chunks_exact_mut(n);
    let (Some(p_r), Some(p_gr), Some(p_gb), Some(p_b)) =
        (planes.next(), planes.next(), planes.next(), planes.next())
    else {
        return Err("internal: output split failed".into());
    };
    let [s_r, s_gr, s_gb, s_b] = phase.slots;
    let [c_r, c_gr, c_gb, c_b] = conv;

    let out_rows = p_r
        .chunks_exact_mut(w2)
        .zip(p_gr.chunks_exact_mut(w2))
        .zip(p_gb.chunks_exact_mut(w2))
        .zip(p_b.chunks_exact_mut(w2));
    for (pair, (((o_r, o_gr), o_gb), o_b)) in src.chunks_exact(2 * w).zip(out_rows) {
        let Some((top, bottom)) = pair.split_at_checked(w) else {
            return Err("internal: row split failed".into());
        };
        // as_chunks drops an odd trailing column.
        let tiles = top.as_chunks::<2>().0.iter().zip(bottom.as_chunks::<2>().0);
        let outs = o_r.iter_mut().zip(o_gr.iter_mut()).zip(o_gb.iter_mut()).zip(o_b.iter_mut());
        for (([tl, tr], [bl, br]), (((d_r, d_gr), d_gb), d_b)) in tiles.zip(outs) {
            let (tl, tr, bl, br) = (*tl, *tr, *bl, *br);
            *d_r = c_r(s_r.pick(tl, tr, bl, br));
            *d_gr = c_gr(s_gr.pick(tl, tr, bl, br));
            *d_gb = c_gb(s_gb.pick(tl, tr, bl, br));
            *d_b = c_b(s_b.pick(tl, tr, bl, br));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGGB: [u8; 4] = [0, 1, 1, 2];

    fn ident(v: u16) -> u16 {
        v
    }

    #[test]
    fn odd_trailing_row_and_column_are_dropped() {
        // 3 rows x 5 cols, value = row * 10 + col.
        let src: Vec<u16> = (0..3).flat_map(|r| (0..5).map(move |c| r * 10 + c)).collect();
        let phase = Phase::from_cfa([2, 2], &RGGB, &[0, 1, 2]).unwrap();
        let mut dst = vec![0u16; 4 * 2];
        pack(&src, 5, &phase, &mut dst, &[ident; 4]).unwrap();
        // R = (0,0),(0,2); Gr = (0,1),(0,3); Gb = (1,0),(1,2); B = (1,1),(1,3).
        assert_eq!(dst, [0, 2, 1, 3, 10, 12, 11, 13]);
    }

    #[test]
    fn too_small_and_wrong_dst_are_errors_not_panics() {
        let phase = Phase::from_cfa([2, 2], &RGGB, &[]).unwrap();
        let mut dst = vec![0u16; 4];
        assert!(pack(&[1, 2, 3], 3, &phase, &mut dst, &[ident; 4]).is_err()); // 1 row
        assert!(pack(&[1, 2, 3], 0, &phase, &mut dst, &[ident; 4]).is_err()); // width 0
        assert!(pack(&[1, 2, 3, 4, 5], 2, &phase, &mut dst, &[ident; 4]).is_err()); // ragged
        assert!(pack(&[1, 2, 3, 4], 2, &phase, &mut dst[..3], &[ident; 4]).is_err());
        assert!(pack(&[1, 2, 3, 4], 2, &phase, &mut dst, &[ident; 4]).is_ok());
    }

    #[test]
    fn plane_color_remaps_pattern_indices() {
        // Pattern indices 0/1/2 mapped to colours B/G/R: [0,1,1,2] reads as BGGR.
        let phase = Phase::from_cfa([2, 2], &RGGB, &[2, 1, 0]).unwrap();
        assert_eq!(
            phase.slots,
            [Slot::BottomRight, Slot::BottomLeft, Slot::TopRight, Slot::TopLeft]
        );
        assert!(Phase::from_cfa([2, 2], &RGGB, &[0, 1, 9]).is_err());
        assert!(Phase::from_cfa([2, 4], &RGGB, &[]).is_err());
    }
}
