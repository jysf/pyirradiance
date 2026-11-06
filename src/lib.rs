//! Prototype Python bindings for `irradiance`.
//!
//! Every entry point takes `bytes` (or an array), returns a NumPy array, and
//! maps every failure to a Python `ValueError` — no panic path of our own.
use irradiance::ifd::Container;
use numpy::ndarray::Dimension;
use numpy::{
    Element, PyArray, PyArray2, PyArray3, PyArrayMethods, PyReadwriteArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

mod pack;

/// A failure inside GIL-released code: the library's typed error, or ours.
enum GlueError {
    Lib(irradiance::Error),
    Invalid(String),
}

impl From<irradiance::Error> for GlueError {
    fn from(e: irradiance::Error) -> Self {
        GlueError::Lib(e)
    }
}

impl From<GlueError> for PyErr {
    fn from(e: GlueError) -> Self {
        match e {
            GlueError::Lib(e) => err(e),
            GlueError::Invalid(msg) => PyValueError::new_err(msg),
        }
    }
}

fn err(e: irradiance::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Downcast a caller-supplied `out=` object, naming the expected array type in
/// the `TypeError` (PyO3's default message says only `PyArray<T, D>`).
fn out_array<'py, T: Element, D: Dimension>(
    out: Bound<'py, PyAny>,
    expected: &str,
) -> PyResult<Bound<'py, PyArray<T, D>>> {
    out.downcast_into::<PyArray<T, D>>()
        .map_err(|_| PyTypeError::new_err(format!("out must be a {expected} numpy array")))
}

/// Borrow a caller-supplied `out=` array for writing.
///
/// Requires C order explicitly: numpy's `as_slice_mut` also accepts
/// Fortran-contiguous arrays, which would be filled transposed.
fn writable<'py, T: Element, D: Dimension>(
    a: &Bound<'py, PyArray<T, D>>,
) -> PyResult<PyReadwriteArray<'py, T, D>> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err("out must be a C-contiguous array"));
    }
    a.try_readwrite()
        .map_err(|e| PyValueError::new_err(format!("out is not writable: {e}")))
}

/// Sensor metadata needed to interpret the mosaic.
#[pyclass(get_all, frozen)]
struct SensorMeta {
    width: u32,
    height: u32,
    bits_per_sample: u32,
    black_level: Option<u32>,
    white_level: Option<u32>,
    /// CFA colour indices row-major (0=R,1=G,2=B), e.g. [0,1,1,2] for RGGB.
    /// `u16`, not `u8`: PyO3 turns `Vec<u8>` into `bytes`, we want `list[int]`.
    cfa_pattern: Option<Vec<u16>>,
    /// `CFARepeatPatternDim` as `(rows, cols)`.
    cfa_repeat_dim: Option<(u32, u32)>,
    /// `CFAPlaneColor`: colour of each pattern index (DNG default `[0, 1, 2]`).
    cfa_plane_color: Option<Vec<u16>>,
    /// Every `BlackLevel` value, row-major over `black_level_repeat_dim`.
    black_levels: Vec<u32>,
    /// `BlackLevelRepeatDim` as `(rows, cols)`; `None` means `(1, 1)`.
    black_level_repeat_dim: Option<(u32, u32)>,
    /// `ActiveArea` as `(top, left, bottom, right)`, bottom/right exclusive.
    active_area: Option<(u32, u32, u32, u32)>,
}

fn parse(data: &[u8]) -> Result<(Container<'_>, irradiance::ifd::Sensor), irradiance::Error> {
    let container = Container::parse(data)?;
    let sensor = container.sensor()?;
    Ok((container, sensor))
}

/// Bound the plane allocation by the input size before allocating it.
///
/// `unpack_into` validates strips only once the caller has allocated `dst`, so
/// a forged `ImageWidth`/`ImageLength` would otherwise request an unbounded
/// buffer and abort on OOM (uncatchable from Python). For an uncompressed
/// plane every pixel's bits must live in the strips, and the strips in `data`,
/// so this caps the plane at `data.len() * 8` samples. Stopgap until upstream
/// ships `plane::validate` (irradiance SPEC-039); `unpack_into` still does
/// the exact checks afterwards.
fn plane_len(sensor: &irradiance::ifd::Sensor, data: &[u8]) -> Result<usize, irradiance::Error> {
    // Compression first, as unpack_into does: a compressed plane is
    // legitimately smaller than w*h*bits, and "unsupported compression" is the
    // error users need to see, not a size mismatch.
    sensor.require_uncompressed()?;
    let file_len = data.len() as u64;
    let mut strip_bytes: u64 = 0;
    for (&offset, &count) in sensor.strip_offsets.iter().zip(&sensor.strip_byte_counts) {
        let end = u64::from(offset) + u64::from(count);
        if end > file_len {
            return Err(irradiance::Error::Truncated {
                at: offset as usize,
                len: count as usize,
            });
        }
        strip_bytes += u64::from(count);
    }
    // Real files never overlap strips; rejecting it stops N overlapping
    // strips from multiplying the bound by N.
    if strip_bytes > file_len {
        return Err(irradiance::Error::Truncated { at: 0, len: strip_bytes as usize });
    }
    // max(bits, 1): a zero bit depth must not void the bound. unpack_into
    // rejects unsupported depths with its own typed error afterwards.
    let bits = u64::from(sensor.bits_per_sample.max(1));
    let strip_bits = strip_bytes * 8;
    let pixels = u64::from(sensor.width) * u64::from(sensor.height); // u32*u32 fits u64
    match pixels.checked_mul(bits) {
        Some(expected_bits) if expected_bits <= strip_bits => {
            // Bounded by data.len() * 8, so it fits usize wherever data does.
            usize::try_from(pixels).map_err(|_| irradiance::Error::PackedSizeMismatch {
                expected_bits,
                strip_bits,
            })
        }
        other => Err(irradiance::Error::PackedSizeMismatch {
            expected_bits: other.unwrap_or(u64::MAX),
            strip_bits,
        }),
    }
}

fn unpack(data: &[u8]) -> Result<(irradiance::ifd::Sensor, Vec<u16>), irradiance::Error> {
    let (container, sensor) = parse(data)?;
    let n = plane_len(&sensor, data)?;
    let mut plane = vec![0u16; n];
    irradiance::plane::unpack_into(&sensor, container.byte_order(), data, &mut plane)?;
    Ok((sensor, plane))
}

/// Unpack into a caller-owned `[H, W]` buffer whose shape is `dst_shape`.
fn unpack_to(
    data: &[u8],
    dst: &mut [u16],
    dst_shape: &[usize],
) -> Result<irradiance::ifd::Sensor, GlueError> {
    let (container, sensor) = parse(data)?;
    plane_len(&sensor, data)?;
    let want = [sensor.height as usize, sensor.width as usize];
    if dst_shape != want {
        return Err(GlueError::Invalid(format!(
            "out has shape {dst_shape:?} but the plane is {want:?}"
        )));
    }
    irradiance::plane::unpack_into(&sensor, container.byte_order(), data, dst)?;
    Ok(sensor)
}

fn meta(s: &irradiance::ifd::Sensor) -> SensorMeta {
    SensorMeta {
        width: s.width,
        height: s.height,
        bits_per_sample: s.bits_per_sample,
        black_level: s.black_level,
        white_level: s.white_level,
        cfa_pattern: s
            .color_filter
            .as_ref()
            .map(|c| c.pattern.iter().map(|&v| u16::from(v)).collect()),
        cfa_repeat_dim: s.color_filter.as_ref().map(|c| (c.repeat_dim[0], c.repeat_dim[1])),
        cfa_plane_color: s
            .color_filter
            .as_ref()
            .map(|c| c.plane_color.iter().map(|&v| u16::from(v)).collect()),
        black_levels: s.black_levels.clone(),
        black_level_repeat_dim: s.black_level_repeat_dim.map(|[r, c]| (r, c)),
        active_area: s.active_area.map(|a| (a.top, a.left, a.bottom, a.right)),
    }
}

/// The black level under each output channel's tile slot.
///
/// The `BlackLevel` pattern's origin is the `ActiveArea` top-left (DNG spec),
/// so the slot parity is shifted by that origin. Only repeat dims that divide
/// the 2x2 tile give one black level per channel; anything else is refused.
fn channel_blacks(m: &SensorMeta, phase: &pack::Phase) -> Result<[f32; 4], String> {
    let (rd, cd) = m.black_level_repeat_dim.unwrap_or((1, 1));
    if !matches!(rd, 1 | 2) || !matches!(cd, 1 | 2) {
        return Err(format!(
            "BlackLevelRepeatDim {rd}x{cd} does not fit a 2x2 Bayer tile; normalise manually"
        ));
    }
    let levels: Vec<u32> = match (m.black_levels.as_slice(), m.black_level) {
        ([], None) => vec![0; (rd * cd) as usize],
        ([], Some(b)) => vec![b; (rd * cd) as usize],
        (all, _) => all.to_vec(),
    };
    if levels.len() != (rd * cd) as usize {
        return Err(format!(
            "{} BlackLevel values for a {rd}x{cd} repeat; normalise manually",
            levels.len()
        ));
    }
    let (top, left) = m.active_area.map_or((0, 0), |(t, l, _, _)| (t, l));
    let mut out = [0f32; 4];
    for (dst, slot) in out.iter_mut().zip(phase.slots) {
        // (slot - origin) mod repeat, in u32 without underflow: + rd*... keeps it positive.
        let r = (slot.row() + rd * 2 - top % rd) % rd;
        let c = (slot.col() + cd * 2 - left % cd) % cd;
        let level = levels
            .get((r * cd + c) as usize)
            .copied()
            .ok_or("internal: black level index out of range")?;
        // u32 black levels above 2^24 would round; real ones are < 2^16.
        *dst = level as f32;
    }
    Ok(out)
}

/// Pack a Bayer plane into `[4, H/2, W/2]` channels `[R, Gr, Gb, B]`.
///
/// `plane` must be the full `uint16[H, W]` plane from `unpack_bayer` (its
/// shape must match `meta`, since the CFA phase is relative to the plane
/// origin). An odd trailing row/column is dropped. Without `normalize`,
/// returns `uint16` raw values. With `normalize=True`, returns `float32`
/// `(v - black) / (white - black)` per channel, **not clipped** (noise goes
/// below 0, specular highlights above 1); black is the per-site `BlackLevel`
/// (`BlackLevelDeltaH/V` are ignored), white is `WhiteLevel` or
/// `2**bits_per_sample - 1`. The masked border is included; crop with
/// `meta.active_area` (halved) if you need only active pixels. `out`, when
/// given, must be a C-contiguous writable array of the output dtype and shape.
#[pyfunction]
#[pyo3(signature = (plane, meta, out=None, normalize=false))]
fn pack_bayer_4ch<'py>(
    py: Python<'py>,
    plane: Bound<'py, PyAny>,
    meta: &Bound<'py, SensorMeta>,
    out: Option<Bound<'py, PyAny>>,
    normalize: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let m = meta.get();
    let invalid = |msg: String| PyValueError::new_err(msg);

    let plane: Bound<'py, PyArray2<u16>> = plane
        .downcast_into()
        .map_err(|_| PyTypeError::new_err("plane must be a uint16[H, W] numpy array"))?;
    let want = [m.height as usize, m.width as usize];
    if plane.shape() != want {
        return Err(invalid(format!(
            "plane has shape {:?} but meta describes {want:?}; pass the full plane from unpack_bayer",
            plane.shape()
        )));
    }
    if !plane.is_c_contiguous() {
        return Err(invalid("plane must be a C-contiguous array".into()));
    }
    let (Some(pattern), Some((rd, cd))) = (&m.cfa_pattern, m.cfa_repeat_dim) else {
        return Err(invalid("no CFA pattern: this is not a Bayer plane".into()));
    };
    // Back to the u8 indices upstream reads; every value came from a u8, and
    // u8::MAX is an unknown colour that from_cfa rejects anyway.
    let to_u8 = |v: &[u16]| -> Vec<u8> { v.iter().map(|&x| u8::try_from(x).unwrap_or(u8::MAX)).collect() };
    let plane_color = to_u8(m.cfa_plane_color.as_deref().unwrap_or(&[]));
    let phase = pack::Phase::from_cfa([rd, cd], &to_u8(pattern), &plane_color).map_err(invalid)?;
    let (w, out_shape) = (want[1], [4, want[0] / 2, want[1] / 2]);

    let src_ro = plane
        .try_readonly()
        .map_err(|e| invalid(format!("plane is not readable: {e}")))?;
    let src = src_ro.as_slice().map_err(|e| invalid(e.to_string()))?;

    if normalize {
        let white = match m.white_level {
            Some(wl) => wl,
            None if (1..=16).contains(&m.bits_per_sample) => (1u32 << m.bits_per_sample) - 1,
            None => return Err(invalid(format!("no WhiteLevel and bits_per_sample {}", m.bits_per_sample))),
        };
        let blacks = channel_blacks(m, &phase).map_err(invalid)?;
        if blacks.iter().any(|&b| b >= white as f32) {
            return Err(invalid(format!("black levels {blacks:?} are not below white {white}")));
        }
        let conv = blacks.map(|black| move |v: u16| (f32::from(v) - black) / (white as f32 - black));
        run_pack::<f32, _>(py, src, w, &phase, out, out_shape, &conv, "float32[4, H/2, W/2]")
    } else {
        let conv = [|v: u16| v; 4];
        run_pack::<u16, _>(py, src, w, &phase, out, out_shape, &conv, "uint16[4, H/2, W/2]")
    }
}

/// Allocate (or borrow `out`), then pack with the GIL released.
#[allow(clippy::too_many_arguments)]
fn run_pack<'py, T, C>(
    py: Python<'py>,
    src: &[u16],
    w: usize,
    phase: &pack::Phase,
    out: Option<Bound<'py, PyAny>>,
    out_shape: [usize; 3],
    conv: &[C; 4],
    expected: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Element + Default + Clone + Send,
    C: Fn(u16) -> T + Sync,
{
    let invalid = |msg: String| PyValueError::new_err(msg);
    if let Some(out) = out {
        let out: Bound<'py, PyArray3<T>> = out_array(out, expected)?;
        if out.shape() != out_shape {
            return Err(invalid(format!("out has shape {:?}, need {out_shape:?}", out.shape())));
        }
        // try_readwrite also refuses an `out` that overlaps the borrowed plane.
        let mut rw = writable(&out)?;
        let dst = rw.as_slice_mut().map_err(|e| invalid(e.to_string()))?;
        py.allow_threads(|| pack::pack(src, w, phase, dst, conv)).map_err(invalid)?;
        drop(rw);
        return Ok(out.into_any());
    }
    let mut buf = vec![T::default(); out_shape.iter().product()];
    py.allow_threads(|| pack::pack(src, w, phase, &mut buf, conv)).map_err(invalid)?;
    let arr = numpy::PyArray1::from_vec(py, buf)
        .reshape(out_shape)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(arr.into_any())
}

/// Unpack the raw sensor plane: `(uint16[H, W], SensorMeta)`.
///
/// With `out`, writes into that C-contiguous, writable `uint16[H, W]` array
/// (e.g. `tensor.numpy()` of a pinned tensor) and returns it; no allocation.
/// On a decode error `out` may be partly written.
#[pyfunction]
#[pyo3(signature = (data, out=None))]
fn unpack_bayer<'py>(
    py: Python<'py>,
    data: &[u8],
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray2<u16>>, SensorMeta)> {
    if let Some(out) = out {
        let out: Bound<'py, PyArray2<u16>> = out_array(out, "uint16[H, W]")?;
        let shape = out.shape().to_vec();
        let mut rw = writable(&out)?;
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let sensor = py.allow_threads(|| unpack_to(data, dst, &shape))?;
        drop(rw);
        return Ok((out, meta(&sensor)));
    }
    // Release the GIL: DataLoader workers may be threads.
    let (sensor, plane) = py.allow_threads(|| unpack(data)).map_err(err)?;
    let flat = numpy::PyArray1::from_vec(py, plane);
    let arr = flat
        .reshape([sensor.height as usize, sensor.width as usize])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((arr, meta(&sensor)))
}

/// Demosaic to camera-space linear RGB: `(float32[H, W, 3], DEVELOP_VERSION)`.
///
/// Uncorrected (no white balance / colour matrix) — see STAGE-008.
#[pyfunction]
fn demosaic<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyArray3<f32>>, &'static str)> {
    let (sensor, rgb) = py
        .allow_threads(|| -> Result<_, irradiance::Error> {
            let (sensor, plane) = unpack(data)?;
            let mut rgb = vec![0u16; plane.len() * 3];
            irradiance::demosaic::demosaic_into(&sensor, &plane, &mut rgb)?;
            let mut f = vec![0f32; rgb.len()];
            irradiance::tone::to_linear_f32_into(&rgb, &mut f)?;
            Ok((sensor, f))
        })
        .map_err(err)?;
    let arr = numpy::PyArray1::from_vec(py, rgb)
        .reshape([sensor.height as usize, sensor.width as usize, 3])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((arr, irradiance::DEVELOP_VERSION))
}

#[pymodule]
#[pyo3(name = "irradiance")]
fn py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unpack_bayer, m)?)?;
    m.add_function(wrap_pyfunction!(demosaic, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bayer_4ch, m)?)?;
    m.add_class::<SensorMeta>()?;
    m.add("DEVELOP_VERSION", irradiance::DEVELOP_VERSION)?;
    Ok(())
}
