"""Descarga los CSV de MiBici con verificación de certificados HTTPS."""

import argparse
from pathlib import Path
import shutil
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent


def create_ssl_context(ca_file=None):
    context = ssl.create_default_context()
    if ca_file is not None:
        context.load_verify_locations(cafile=str(ca_file))
    else:
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
    return context


def download(url, directory, overwrite=False, attempts=3, timeout=60, context=None):
    destination = directory / Path(urlsplit(url).path).name
    if not overwrite and destination.exists() and destination.stat().st_size > 0:
        return "omitido"
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "MiBici-CSV-Downloader/1.0"})
            with urlopen(request, timeout=timeout, context=context) as response:
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("El servidor devolvió HTML en lugar de un CSV")
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                size = partial.stat().st_size
                expected = response.headers.get("Content-Length")
                if size == 0 or (expected is not None and size != int(expected)):
                    raise ValueError("La descarga está vacía o incompleta")
            partial.replace(destination)
            return "descargado"
        except (OSError, URLError, ValueError) as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts or (
                isinstance(error, HTTPError) and 400 <= error.code < 500
                and error.code not in (408, 429)
            ):
                raise
            time.sleep(2 ** (attempt - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR.parent / "downloads",
        help="Carpeta de destino (por defecto: mi-bici-experiment/downloads)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Volver a descargar existentes")
    parser.add_argument("--ca-file", type=Path, help="Certificados CA adicionales en formato PEM")
    args = parser.parse_args()
    try:
        context = create_ssl_context(args.ca_file)
    except (OSError, ssl.SSLError) as error:
        parser.error(f"No se pudieron cargar los certificados: {error}")
    urls = (SCRIPT_DIR / "mibici_urls.txt").read_text().splitlines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"descargado": 0, "omitido": 0, "fallido": 0}
    for index, url in enumerate(urls, start=1):
        name = Path(urlsplit(url).path).name
        try:
            status = download(url, args.output_dir, args.overwrite, context=context)
            counts[status] += 1
            print(f"[{index}/{len(urls)}] {status}: {name}", flush=True)
        except (OSError, URLError, ValueError) as error:
            counts["fallido"] += 1
            print(f"[{index}/{len(urls)}] fallido: {name}: {error}", file=sys.stderr, flush=True)
    print(f"Resultado: {counts}. Destino: {args.output_dir.resolve()}")
    return 1 if counts["fallido"] else 0


if __name__ == "__main__":
    sys.exit(main())
