import numpy as np

from invoice_router.pipeline.preprocessing import (
    _choose_best_upright_orientation,
    deskew_and_normalize,
    load_image,
)


def test_choose_best_upright_orientation_prefers_rotated_page(monkeypatch):
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    rotated = np.rot90(image, 2)

    def fake_score(candidate):
        if np.array_equal(candidate, rotated):
            return 4.0
        return 1.0

    monkeypatch.setattr("invoice_router.pipeline.preprocessing._readability_score", fake_score)

    result = _choose_best_upright_orientation(image)

    assert np.array_equal(result, rotated)


def test_deskew_and_normalize_keeps_upright_when_rotation_not_better(monkeypatch):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "invoice_router.pipeline.preprocessing._readability_score", lambda _candidate: 2.0
    )

    normalized = deskew_and_normalize(image)

    assert normalized.shape == image.shape


def test_load_image_limits_pdf_pages_to_first_three(monkeypatch, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePixmap:
        def __init__(self):
            self.n = 3
            self.alpha = 0
            self.h = 10
            self.w = 10
            self.samples = np.arange(self.h * self.w * self.n, dtype=np.uint8).tobytes()

    class FakePage:
        def get_pixmap(self, matrix):
            assert matrix is not None
            return FakePixmap()

    class FakeDoc:
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return FakePage()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("invoice_router.pipeline.preprocessing.fitz.open", lambda _path: FakeDoc())

    pages = load_image(str(pdf_path))

    assert len(pages) == 3
    assert all(page.shape == (10, 10, 3) for page in pages)


def test_load_image_resizes_large_image_input(tmp_path):
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"img")

    image = np.zeros((3000, 1200, 3), dtype=np.uint8)
    from invoice_router.pipeline import preprocessing as preprocessing_module

    original_imread = preprocessing_module.cv2.imread
    original_cvt_color = preprocessing_module.cv2.cvtColor
    try:
        preprocessing_module.cv2.imread = lambda _path: image.copy()
        preprocessing_module.cv2.cvtColor = lambda data, _code: data
        pages = load_image(str(image_path))
    finally:
        preprocessing_module.cv2.imread = original_imread
        preprocessing_module.cv2.cvtColor = original_cvt_color

    assert len(pages) == 1
    assert max(pages[0].shape[:2]) == 2048
