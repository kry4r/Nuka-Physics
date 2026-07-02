# Vendored: stb_image / stb_image_write

Single-header public-domain image codecs by Sean Barrett (nothings/stb).

- `stb_image.h` — decode PNG/JPG/TGA/BMP/HDR. Used by `src/render/texture_image.cpp`
  (the single translation unit defining `STB_IMAGE_IMPLEMENTATION`) to load PBR
  texture maps and equirectangular `.hdr` environment maps for the offline beauty
  path tracer.
- `stb_image_write.h` — vendored for completeness (host PNG/HDR writeback); not yet
  compiled into any target.

Source: https://github.com/nothings/stb (master).
License: dual public-domain / MIT (see the license text at the foot of each header).
