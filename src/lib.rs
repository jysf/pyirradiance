//! Prototype Python bindings for `irradiance`.
//!
//! Every entry point takes `bytes`, returns a NumPy array, and maps the
//! library's typed `Error` to a Python `ValueError` — no panic path of our own.
use irradiance::ifd::Container;
use numpy::{PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(e: irradiance::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
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

fn unpack(data: &[u8]) -> Result<(irradiance::ifd::Sensor, Vec<u16>), irradiance::Error> {
    let (container, sensor) = parse(data)?;
    let n = usize::try_from(u64::from(sensor.width) * u64::from(sensor.height)).map_err(|_| {
        irradiance::Error::PlaneBufferWrongLength { expected: u64::MAX, actual: 0 }
    })?;
    let mut plane = vec![0u16; n];
    irradiance::plane::unpack_into(&sensor, container.byte_order(), data, &mut plane)?;
    Ok((sensor, plane))
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
#[pyfunction]
fn unpack_bayer<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyArray2<u16>>, SensorMeta)> {
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
