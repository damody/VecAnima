import cv2
import numpy as np

from vecanima.vectorize import Settings, fit_palette, mask_rings, rasterize, svg, vectorize


def test_hole_thin_line_and_separate_dark_layer():
    image = np.full((64,64,3), 240, np.uint8)
    image[8:56,8:56] = (30,120,210)
    image[24:40,24:40] = 240
    image[15,10:54] = 10
    settings = Settings(colors=3, epsilon=.1)
    palette = fit_palette([image], settings)
    project = vectorize(image,palette,settings)
    text = svg(project)
    _, rendered = rasterize(text)
    assert '<image' not in text
    assert 'id="dark-mask-baseline"' in text
    assert rendered[30,30].min() > 200  # central hole
    assert rendered[15,20].max() < 25  # single-pixel dark feature
    assert rendered[45,45,2] > rendered[45,45,0] + 80  # region color


def test_palette_and_geometry_are_repeatable():
    rng = np.random.default_rng(7)
    image = rng.integers(0,256,(40,40,3),dtype=np.uint8)
    settings = Settings(colors=8)
    a, b = fit_palette([image],settings), fit_palette([image],settings)
    assert np.array_equal(a,b)
    assert svg(vectorize(image,a,settings)) == svg(vectorize(image,b,settings))


def test_solid_black_has_valid_opaque_output():
    image = np.zeros((32,32,3),np.uint8)
    settings = Settings(colors=2)
    p = vectorize(image,fit_palette([image],settings),settings)
    _, rendered = rasterize(svg(p))
    assert rendered.max() == 0


def test_pixel_cell_tracing_preserves_diagonal_contacts_and_nested_holes():
    mask = np.zeros((16,16), np.uint8)
    mask[0:12,0:12] = 255
    mask[2:10,2:10] = 0
    mask[4:8,4:8] = 255
    mask[12,12] = 255  # diagonal contact, separate 4-connected component
    mask[15,15] = 255  # isolated corner pixel
    project = {"width":16,"height":16,"background":"#ffffff", "layers":[
        {"kind":"region","fill":"#000000","rings":mask_rings(mask,0)}]}
    _, rendered = rasterize(svg(project))
    assert np.array_equal(rendered[:,:,0],255-mask)
