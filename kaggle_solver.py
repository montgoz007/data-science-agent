#!/usr/bin/env python3
import os
import subprocess
import zipfile
from dotenv import load_dotenv
import typer

# 1. Load .env right away so KAGGLE_USERNAME & KAGGLE_KEY are in env
load_dotenv()

def main(
    competition: str = typer.Argument(
        ...,
        help="Kaggle competition slug (e.g., titanic)"
    )
):
    """
    Download & unzip all files for a Kaggle competition into data/{competition}.
    """
    out_dir = os.path.join("data", competition)
    os.makedirs(out_dir, exist_ok=True)

    typer.secho(f"⏬ Downloading '{competition}' into '{out_dir}' …", fg=typer.colors.CYAN)
    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition,
        "-p", out_dir
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        typer.secho("❌ Download failed:", fg=typer.colors.RED)
        typer.echo(proc.stderr or proc.stdout)
        raise typer.Exit(code=1)

    # 2. Unzip any .zip files in out_dir
    for fname in os.listdir(out_dir):
        if fname.endswith(".zip"):
            zip_path = os.path.join(out_dir, fname)
            typer.secho(f"📂 Unzipping '{fname}' …", fg=typer.colors.CYAN)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(out_dir)
            os.remove(zip_path)

    typer.secho("✅ Download and unzip complete!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    typer.run(main)