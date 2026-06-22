import zipfile
from urllib.error import HTTPError

from musemotion.music.emopia import _download_with_retries, download_emopia_dataset


def test_download_emopia_dataset_extracts_single_root_archives(tmp_path):
    archive_path = tmp_path / "EMOPIA_1.0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("EMOPIA_1.0/midis/Q1_demo.mid", b"midi")
        archive.writestr("EMOPIA_1.0/label.csv", "clip,quadrant\nQ1_demo.mid,Q1\n")

    output_dir = tmp_path / "emopia"

    result = download_emopia_dataset(output_dir=output_dir, url=str(archive_path))

    assert result == output_dir
    assert (output_dir / "midis" / "Q1_demo.mid").read_bytes() == b"midi"
    assert (output_dir / "label.csv").exists()


def test_download_with_retries_recovers_from_transient_http_error(tmp_path):
    calls = []
    archive_path = tmp_path / "archive.zip"

    def flaky_download(url, filename):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 504, "Gateway Time-out", {}, None)
        archive_path.write_bytes(b"zip")

    _download_with_retries(
        "https://example.test/archive.zip",
        archive_path,
        attempts=2,
        sleep_seconds=0,
        download_fn=flaky_download,
    )

    assert len(calls) == 2
    assert archive_path.read_bytes() == b"zip"
