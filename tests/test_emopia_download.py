import zipfile

from musemotion.music.emopia import download_emopia_dataset


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

