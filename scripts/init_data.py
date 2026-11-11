#!/usr/bin/env python3
"""Download and verify the MovieLens dataset used by the course experiment."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = (
    "https://box.nju.edu.cn/d/5366547b75e549ac814b/files/"
    "?p=%2Fml-1m.zip&dl=1"
)
ARCHIVE_SHA256 = "8a0072924e7578a8251e3a1783bafdaad848a7a309ea39160eaf22966e78657b"
# Expected files for input version ml-1m-local-raw-v1.
FILE_SHA256 = {
    "README": "2d5ddce86ccbd247207456165827b4b9ddf27282bca93807d4c763e28414f133",
    "users.dat": "c0397b2eb6210f14a683c65d15d7d927aaffac0c08bcbefddb8cc2063a988279",
    "movies.dat": "e256099e0a8f2d7cdc4b97ac94864c5f05795b929a97b045516ac86303931e18",
    "ratings.dat": "1b3c4690b619d5f6bead9283d02ad12460da88bd3087ac2a93959bb37f4995d2",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(directory: Path) -> None:
    for name, expected in FILE_SHA256.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} 缺失或与实验数据版本 ml-1m-local-raw-v1 不一致。")


def initialize(directory: Path) -> None:
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"数据路径不是目录：{directory}")
        if any(directory.iterdir()):
            try:
                verify_files(directory)
            except ValueError as exc:
                raise ValueError(f"{exc} 已保留现有文件，请用 --data-dir 指定一个空目录。") from exc
            print(f"MovieLens 实验数据校验通过，数据已就绪：{directory}")
            return

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ml-1m-", dir=directory.parent) as temporary:
        archive = Path(temporary) / "ml-1m.zip"
        print(f"正在下载：{DATA_URL}", flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("下载文件的 SHA-256 校验失败，请重新下载。")

        extracted = Path(temporary) / "ml-1m"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for name in FILE_SHA256:
                (extracted / name).write_bytes(bundle.read(f"ml-1m/{name}"))
        verify_files(extracted)
        if directory.exists():
            directory.rmdir()
        extracted.rename(directory)
    print(f"MovieLens 实验数据初始化完成：{directory}")
    print("用户记录 6,946 条；电影记录 4,465 条；评分记录 1,150,241 条。使用条件见数据目录中的 README。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "ml-1m",
                        help="数据目录，默认为项目根目录下的 ml-1m")
    args = parser.parse_args()
    try:
        initialize(args.data_dir.expanduser().resolve())
    except (OSError, urllib.error.URLError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
