# Viewer font sources

The following original font files are fetched at CMake configuration time and
copied beside the viewer at build time. No font binary is stored in this folder.

- Inter Regular and SemiBold: unmodified `extras/ttf` files from the official [Inter 4.1 release](https://github.com/rsms/inter/releases/tag/v4.1), licensed under the included Inter-OFL.txt
  - Archive: `https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip`
  - Archive SHA256: `9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e`
  - Regular SHA256: `40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82`
  - SemiBold SHA256: `78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3`
- Noto Sans CJK SC Regular: [official revision f8d1575](https://github.com/notofonts/noto-cjk/blob/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf), licensed under the included NotoSansCJK-OFL.txt
  - SHA256: `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`

For offline font provisioning, set all three CMake FILEPATH options to the exact
original files: `NUKA_VIEWER_INTER_REGULAR_FONT`, `NUKA_VIEWER_INTER_SEMIBOLD_FONT`
and `NUKA_VIEWER_CJK_FONT`. Other build dependencies must already be available.
Missing or mismatched files stop configuration; there is no silent font substitution.
The default download cache is `<build>/_deps/nuka-fonts`; both the Inter archive
and extracted files are hash-checked. Unset Inter overrides download the archive
once, while supplying both Inter overrides avoids the archive entirely.

The viewer loads packaged resources without network access. It does not install
system fonts. Inter Regular is also embedded at build time as a proportional UI
fallback. Existing JetBrains Mono resources remain reserved for numeric and log
roles. Runtime loading does not repeat the configuration-time hash checks.
