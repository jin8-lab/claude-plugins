#!/usr/bin/env python3
"""Gemini の画像生成 API を呼び、画像をファイルに保存する。

使い方:
  python3 generate.py -p "prompt" -o out.png [-a 16:9] [-r ref1.png -r ref2.png] [-m model]

APIキー:
  - クラウドセッション: 中継側 (Network secrets) が x-goog-api-key を付けるので不要。
  - 手元の PC: 環境変数 GEMINI_API_KEY に入れる。
モデル: -m か環境変数 GEMINI_IMAGE_MODEL。既定は DEFAULT_MODEL。
"""
import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "gemini-2.5-flash-image"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-a", "--aspect", default="1:1", choices=RATIOS)
    ap.add_argument("-r", "--ref", action="append", default=[], help="画風・構図の参考画像（複数可）")
    ap.add_argument("-m", "--model", default=os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL))
    a = ap.parse_args()

    parts = [{"text": a.prompt}]
    for path in a.ref:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(f.read()).decode()}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": a.aspect},
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(body, tf)
        body_path = tf.name
    try:
        # curl は環境のプロキシ設定をそのまま使える
        key = os.environ.get("GEMINI_API_KEY", "dummy")
        r = subprocess.run(
            ["curl", "-sS", "-X", "POST", ENDPOINT.format(model=a.model),
             "-H", "Content-Type: application/json",
             "-H", f"x-goog-api-key: {key}",
             "--data-binary", f"@{body_path}"],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(body_path)

    if r.returncode != 0:
        sys.exit(f"curl failed: {r.stderr.strip()}")
    try:
        res = json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.exit(f"API の応答が JSON ではありません:\n{r.stdout[:2000]}")
    if "error" in res:
        err = res["error"]
        hint = ""
        if err.get("code") == 429 and "free_tier" in err.get("message", ""):
            hint = ("\n\n[ヒント] 無料枠では画像生成の割り当てが 0 です。Google AI Studio "
                    "(https://aistudio.google.com/apikey) でこのキーのプロジェクトに"
                    "お支払い情報を設定すると使えるようになります。")
        sys.exit(f"API error: {err.get('code')} {err.get('status')}: {err.get('message', '')[:400]}{hint}")

    texts, image = [], None
    for cand in res.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            data = part.get("inlineData") or part.get("inline_data")
            if data and image is None:
                image = data
            elif "text" in part:
                texts.append(part["text"])
    if image is None:
        reason = [c.get("finishReason") for c in res.get("candidates", [])]
        sys.exit(f"画像が返りませんでした。finishReason={reason} "
                 f"promptFeedback={res.get('promptFeedback')} text={' '.join(texts)[:500]}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(base64.b64decode(image["data"]))
    # あとで同じ画風で作り直せるよう、プロンプトを残す
    with open(a.out + ".prompt.txt", "w") as f:
        f.write(f"model: {a.model}\naspect: {a.aspect}\nrefs: {', '.join(a.ref) or '-'}\n\n{a.prompt}\n")

    print(f"saved: {a.out} ({image.get('mimeType') or image.get('mime_type')})")
    if texts:
        print("model note:", " ".join(texts)[:500])


if __name__ == "__main__":
    main()
