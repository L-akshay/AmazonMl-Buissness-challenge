"""Extract organizer files safely, preserving ZIP originals and excluding metadata."""

import argparse
import json
from pathlib import Path
import shutil
import zipfile


def extract(archive, destination, prefix):
    destination = destination.resolve()
    records = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = Path(member.filename[len(prefix):])
            if any(p.startswith(".") or p == "__MACOSX" for p in relative.parts):
                continue
            if prefix and relative.as_posix() in ("README.md", "Documentation_template.md"):
                relative = Path("reference") / relative
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if target.exists():
                raise FileExistsError(f"Refusing to replace {target}; choose a fresh destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            records.append({"archive": str(archive.resolve()), "member": member.filename,
                            "path": str(target), "bytes": member.file_size, "crc32": f"{member.CRC:08x}"})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--brief-zip", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract(args.data_zip, args.destination, "student_resource/")
    if args.brief_zip:
        manifest += extract(args.brief_zip, args.destination / "brief", "")
    reports = args.destination / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
