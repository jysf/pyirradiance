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
    cfa_pattern: Option<Vec<u8>>,
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
        cfa_pattern: s.color_filter.as_ref().map(|c| c.pattern.clone()),
    }
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
    m.add_class::<SensorMeta>()?;
    m.add("DEVELOP_VERSION", irradiance::DEVELOP_VERSION)?;
    Ok(())
}
