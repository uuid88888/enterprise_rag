"""OCR 配置与可选实测脚本。

默认只做本地配置检查，不调用云端模型。
如需真实调用 OCR：
    python ocr_selftest.py --image path/to/image.png
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ocr import get_ocr_provider  # noqa: E402
from utils.config import settings  # noqa: E402


def _mask_state(value: str) -> str:
    return "已配置" if value else "未配置"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 OCR 配置，可选对图片发起真实 OCR 调用。")
    parser.add_argument("--image", help="可选：提供图片路径后发起真实 OCR 调用")
    args = parser.parse_args()

    print("OCR 配置检查")
    print(f"  ENABLE_OCR       : {settings.enable_ocr}")
    print(f"  OCR_PROVIDER     : {settings.ocr_provider}")
    print(f"  OCR_MODEL        : {settings.ocr_model}")
    print(f"  OCR_BASE_URL     : {settings.ocr_base_url}")
    print(f"  OCR_API_KEY      : {_mask_state(settings.effective_ocr_api_key)}")
    print(f"  OCR_MAX_PAGES    : {settings.ocr_max_pages}")
    print(f"  OCR_DPI          : {settings.ocr_dpi}")

    if not settings.enable_ocr:
        print("\n结果：OCR 当前未启用。设置 ENABLE_OCR=true 后可处理图片/扫描版 PDF。")
        return 0
    if not settings.effective_ocr_api_key:
        print("\n结果：OCR 已启用，但没有可用 API Key。")
        return 1

    provider = get_ocr_provider()
    print(f"\nProvider 初始化成功：{provider.__class__.__name__}")

    if not args.image:
        print("未传入 --image，跳过真实云端 OCR 调用。")
        return 0

    if not os.path.exists(args.image):
        print(f"图片不存在：{args.image}")
        return 1

    mime_type = mimetypes.guess_type(args.image)[0] or "image/png"
    with open(args.image, "rb") as f:
        text = provider.extract_image_bytes(f.read(), mime_type=mime_type)
    print("\nOCR 返回文本：")
    print(text or "（空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
