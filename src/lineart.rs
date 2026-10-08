//! Optional local AniLines RGB ONNX inference. No Python or remote inference.
use anyhow::{Context, Result, ensure};
use opencv::{core, dnn, prelude::*};
use std::{cell::RefCell, path::Path};

thread_local! {
    static NETWORK: RefCell<Option<(String, dnn::Net)>> = const { RefCell::new(None) };
}

pub fn identity(path: Option<&Path>) -> Result<Option<serde_json::Value>> {
    path.map(|p| {
        Ok(
            serde_json::json!({"path":p.canonicalize().context("Cannot open line model")?,
            "sha256":crate::cache::file(p)?}),
        )
    })
    .transpose()
}

pub fn predict(image: &core::Mat, path: &Path) -> Result<Vec<f64>> {
    // Bound DNN activation memory for arbitrary-resolution inputs. Geometry and
    // colors still use the original working image, only boundary evidence scales.
    let pixels = image.total() as f64;
    let scale = (262144. / pixels)
        .sqrt()
        .min(768. / image.cols().max(image.rows()) as f64)
        .min(1.);
    if scale < 1. {
        let mut reduced = core::Mat::default();
        opencv::imgproc::resize(
            image,
            &mut reduced,
            core::Size::new(
                (image.cols() as f64 * scale).round().max(1.) as i32,
                (image.rows() as f64 * scale).round().max(1.) as i32,
            ),
            0.,
            0.,
            opencv::imgproc::INTER_AREA,
        )?;
        let hint = predict_inner(&reduced, path)?;
        let data = core::Mat::from_slice(&hint)?;
        let map = data.reshape(1, reduced.rows())?;
        let mut expanded = core::Mat::default();
        opencv::imgproc::resize(
            &map,
            &mut expanded,
            image.size()?,
            0.,
            0.,
            opencv::imgproc::INTER_LINEAR,
        )?;
        return Ok(expanded.data_typed::<f64>()?.to_vec());
    }
    predict_inner(image, path)
}

fn predict_inner(image: &core::Mat, path: &Path) -> Result<Vec<f64>> {
    let digest = crate::cache::file(path)?;
    let (w, h) = (image.cols() as usize, image.rows() as usize);
    let (pw, ph) = (w.div_ceil(16) * 16, h.div_ceil(16) * 16);
    let source = image.data_typed::<core::Vec3b>()?;
    let mut rgb = vec![core::Vec3f::default(); pw * ph];
    // Pillow SMOOTH + ImageEnhance.Sharpness(6), used by the official basic model.
    let reflect = |p: usize, n: usize| -> usize {
        if n == 1 {
            0
        } else {
            let q = p % (2 * n - 2);
            if q < n { q } else { 2 * n - 2 - q }
        }
    };
    for y in 0..ph {
        for x in 0..pw {
            let (sx, sy) = (reflect(x, w), reflect(y, h));
            for c in 0..3 {
                let original = source[sy * w + sx][2 - c] as i32;
                let value = if sx > 0 && sy > 0 && sx + 1 < w && sy + 1 < h {
                    let mut sum = 4 * original;
                    for yy in sy - 1..=sy + 1 {
                        for xx in sx - 1..=sx + 1 {
                            sum += source[yy * w + xx][2 - c] as i32;
                        }
                    }
                    (6 * original - 5 * ((sum + 6) / 13)).clamp(0, 255)
                } else {
                    original
                };
                rgb[y * pw + x][c] = value as f32 / 255.;
            }
        }
    }
    let data = core::Mat::from_slice(&rgb)?;
    let padded = data.reshape(3, ph as i32)?;
    let blob = dnn::blob_from_image(
        &padded,
        1.,
        core::Size::default(),
        core::Scalar::default(),
        false,
        false,
        core::CV_32F,
    )?;
    NETWORK.with(|state| -> Result<Vec<f64>> {
        let mut state = state.borrow_mut();
        if state.as_ref().is_none_or(|(key, _)| key != &digest) {
            let bytes = std::fs::read(path)?;
            let mut net = dnn::read_net_from_onnx_buffer(&core::Vector::from_slice(&bytes))
                .context("Cannot load AniLines RGB ONNX model")?;
            net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
            net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
            *state = Some((digest, net));
        }
        let net = &mut state.as_mut().unwrap().1;
        net.set_input_def(&blob)?;
        let result = net.forward_single_def().context(
            "Line model must accept NCHW RGB and emit a white-background single-channel line map",
        )?;
        ensure!(
            result.total() == pw * ph && result.channels() == 1,
            "Invalid line-model output shape"
        );
        let pixels = result.data_typed::<f32>()?;
        ensure!(
            pixels.iter().all(|p| p.is_finite()),
            "Non-finite line-model output"
        );
        Ok((0..h)
            .flat_map(|y| {
                let pixels = &pixels;
                (0..w).map(move |x| (1. - pixels[y * pw + x] as f64).clamp(0., 1.))
            })
            .collect())
    })
}
