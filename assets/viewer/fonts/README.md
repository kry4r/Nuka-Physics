# Viewer font sources

The following original font files are fetched at CMake configuration time and
copied beside the viewer at build time. No font binary is stored in this folder.

- Nerd Fonts v3.4.0 release archives from the [official release](https://github.com/ryanoasis/nerd-fonts/releases/tag/v3.4.0);
  the editor UI uses their text faces, and its icons come from Phosphor below.
  - `UbuntuSans.tar.xz` (`UbuntuSansNerdFontPropo-Regular.ttf`, `-Medium.ttf`; Ubuntu Font Licence, copied as `UbuntuSans-LICENCE.txt`)
    - SHA256: `0a399b03565c577ae2da9f8362e6276b0adb13bac288e49fec085c517243e27b`
  - `MartianMono.tar.xz` (`MartianMonoNerdFontPropo-Medium.ttf`; SIL OFL 1.1, copied as `MartianMono-OFL.txt`)
    - SHA256: `e12f5ca02756268e794d06d9cb2ada93d021ec915f5a43ec2cd77355022f3624`
  - `GeistMono.tar.xz` (`GeistMonoNerdFontPropo-Regular.otf`; SIL OFL 1.1, copied as `GeistMono-OFL.txt`)
    - SHA256: `43ef3d73cadddea5be746e20c9f27fbaa59ceb106b8b26f7245ba2de2e46b1f6`
- Noto Sans CJK SC Regular: [official revision f8d1575](https://github.com/notofonts/noto-cjk/blob/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf), licensed under the included NotoSansCJK-OFL.txt
  - SHA256: `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`
- Phosphor Icons 2.1.1 from the [`@phosphor-icons/web` npm package](https://www.npmjs.com/package/@phosphor-icons/web/v/2.1.1)
  (MIT, copied as `Phosphor-LICENSE.txt`); the regular face is merged into every text face, the fill face draws active states.
  - `src/regular/Phosphor.ttf` SHA256: `06b91e022b7ee899a63efced879392a74f0bacbda54e4467e9f663220d173a10`
  - `src/fill/Phosphor-Fill.ttf` SHA256: `a53f5d2630cab5e3b7536ecb9d69d71519a2190298c22b1f8d770dd37bc2940a`
  - `LICENSE` SHA256: `81bb1bfd8e3730c19a3895ff4a82839980933504725960c52b1c207b29398f07`

For offline provisioning, set `NUKA_VIEWER_NERD_FONT_DIR` to a directory holding
the three archives above, `NUKA_VIEWER_CJK_FONT` to the exact CJK file and
`NUKA_VIEWER_ICON_FONT_DIR` to a directory holding `Phosphor.ttf`, `Phosphor-Fill.ttf` and `LICENSE`.
Missing or mismatched files stop configuration; there is no silent font substitution.
The default download cache is `<build>/_deps/nuka-fonts`; archives are hash-checked
before the needed faces and licences are extracted.

The viewer loads packaged resources without network access and does not install
system fonts. Runtime loading does not repeat the configuration-time hash checks.
