<br>

<p align="center">
	<img width="300" src="assets/icon.png" alt="logo">
	<br>
	<h2 align="center">Image Deduplication &nbsp;<img alt="Static Badge" src="https://img.shields.io/badge/Experiment-e28743"></h2>
	<p align="center">A clustering algorithm example for de-duplicating near exact images in videos using vector embeddings.</p>
	<p align="center">
		<a href="https://github.com/codespaces/new/HQarroum/image-deduplication"><img alt="Github Codespaces" src="https://github.com/codespaces/badge.svg" /></a>
	</p>
</p>
<br>

## 🔖 Features

- 📹 **Scene Detection** - Uses scene detection algorithms from [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to extract transition frames from videos.
- 🤖 **Semantic Fingerprinting** — Use vector embeddings to perform semantic de-duplication of images.
- ⬛ **Technical Frames Detection** — Detects black and white frames in videos and removes them from the set.
- 🖼️ **Blurry Frame Detection** — Detects blurry frames in videos using Laplacian variance and removes them from the set.

## 🚀 Installation

**Using `pip`**

```bash
$ pip install -r requirements.txt
```

**Using `uv`**

```bash
$ uv sync
```

> This application requires ffmpeg/mkvmerge for video splitting support.

## What's this ❓
