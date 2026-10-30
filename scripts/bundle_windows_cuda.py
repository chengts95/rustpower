"""Add the small, dynamically loaded CUDA bridge to a Windows wheel.

NVIDIA runtime DLLs deliberately stay in the optional nvidia-cudss-cu12 pip
package. Keeping them out of this wheel preserves a normal CPU installation.
"""
import argparse
import base64
import csv
import hashlib
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, default=Path("target/release"))
    args = parser.parse_args()
    bridge = args.native_dir / "acpf_cuda.dll"
    if not bridge.is_file():
        raise SystemExit(f"CUDA bridge not found: {bridge}")

    temporary = args.wheel.with_suffix(".cuda.tmp.whl")
    payload = bridge.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    bridge_name = "rustpower/acpf_cuda.dll"
    with zipfile.ZipFile(args.wheel, "r") as source, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        record_name = next(
            (name for name in source.namelist() if name.endswith(".dist-info/RECORD")), None
        )
        if record_name is None:
            raise SystemExit("wheel has no .dist-info/RECORD")
        for info in source.infolist():
            if info.filename in {"acpf_cuda.dll", bridge_name}:
                continue
            data = source.read(info.filename)
            if info.filename == record_name:
                rows = [
                    row
                    for row in csv.reader(data.decode("utf-8").splitlines())
                    if row and row[0] not in {"acpf_cuda.dll", bridge_name}
                ]
                rows.append([bridge_name, f"sha256={digest}", str(len(payload))])
                import io

                output = io.StringIO(newline="")
                csv.writer(output, lineterminator="\r\n").writerows(rows)
                data = output.getvalue().encode("utf-8")
            target.writestr(info, data)
        target.writestr(bridge_name, payload)
    temporary.replace(args.wheel)
    print(f"Added {bridge.name} ({len(payload):,} bytes) to {args.wheel}")


if __name__ == "__main__":
    main()
