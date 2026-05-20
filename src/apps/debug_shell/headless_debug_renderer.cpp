// ---------------------------------------------------------------------------
// nuka::app::headless_debug_renderer implementation
// ---------------------------------------------------------------------------

#include "apps/debug_shell/headless_debug_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace nuka::app {

namespace {

struct PixelPoint {
    int x = 0;
    int y = 0;
};

Rgba8 ColorFromCommand(uint32_t color) {
    return {
        static_cast<uint8_t>((color >> 24) & 0xFFu),
        static_cast<uint8_t>((color >> 16) & 0xFFu),
        static_cast<uint8_t>((color >> 8) & 0xFFu),
        static_cast<uint8_t>(color & 0xFFu)
    };
}

PixelPoint Project(math::Vec3 point, const DebugRasterOptions& options) {
    const float x = (point.x - options.view_center.x) * options.view_scale
        + static_cast<float>(options.width) * 0.5f;
    const float y = static_cast<float>(options.height) * 0.5f
        - (point.y - options.view_center.y) * options.view_scale;
    return {
        static_cast<int>(std::lround(x)),
        static_cast<int>(std::lround(y))
    };
}

void PutPixel(DebugRasterImage& image, int x, int y, Rgba8 color) {
    if (x < 0 || y < 0) {
        return;
    }
    const auto ux = static_cast<uint32_t>(x);
    const auto uy = static_cast<uint32_t>(y);
    if (ux >= image.width || uy >= image.height) {
        return;
    }
    image.pixels[static_cast<size_t>(uy) * image.width + ux] = color;
}

void DrawDisk(DebugRasterImage& image, PixelPoint center, int radius, Rgba8 color) {
    const int r = std::max(radius, 1);
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {
            if ((x * x + y * y) <= (r * r)) {
                PutPixel(image, center.x + x, center.y + y, color);
            }
        }
    }
}

void DrawLine(DebugRasterImage& image, PixelPoint a, PixelPoint b, Rgba8 color) {
    int x0 = a.x;
    int y0 = a.y;
    const int x1 = b.x;
    const int y1 = b.y;

    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
        PutPixel(image, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

void DrawRect(DebugRasterImage& image,
              PixelPoint center,
              int half_width,
              int half_height,
              Rgba8 color) {
    const int hw = std::max(half_width, 1);
    const int hh = std::max(half_height, 1);
    const PixelPoint a{center.x - hw, center.y - hh};
    const PixelPoint b{center.x + hw, center.y - hh};
    const PixelPoint c{center.x + hw, center.y + hh};
    const PixelPoint d{center.x - hw, center.y + hh};
    DrawLine(image, a, b, color);
    DrawLine(image, b, c, color);
    DrawLine(image, c, d, color);
    DrawLine(image, d, a, color);
}

void RasterizeCommand(DebugRasterImage& image,
                      const DrawCommand& command,
                      const DebugRasterOptions& options) {
    const Rgba8 color = ColorFromCommand(command.color);
    const PixelPoint position = Project(command.position, options);

    switch (command.type) {
    case DrawCommandType::Line:
        DrawLine(image, position, Project(command.end, options), color);
        break;
    case DrawCommandType::Sphere:
        DrawDisk(image, position, static_cast<int>(std::ceil(command.radius * options.view_scale)), color);
        break;
    case DrawCommandType::Capsule: {
        const math::Vec3 axis = command.end.Normalized();
        const math::Vec3 a = command.position - axis * command.half_height;
        const math::Vec3 b = command.position + axis * command.half_height;
        DrawLine(image, Project(a, options), Project(b, options), color);
        DrawDisk(image, Project(a, options), static_cast<int>(std::ceil(command.radius * options.view_scale)), color);
        DrawDisk(image, Project(b, options), static_cast<int>(std::ceil(command.radius * options.view_scale)), color);
        break;
    }
    case DrawCommandType::Box:
        DrawRect(image,
                 position,
                 static_cast<int>(std::ceil(command.size.x * options.view_scale)),
                 static_cast<int>(std::ceil(command.size.y * options.view_scale)),
                 color);
        break;
    case DrawCommandType::AABB: {
        const PixelPoint min = Project(command.position, options);
        const PixelPoint max = Project(command.end, options);
        const PixelPoint center{(min.x + max.x) / 2, (min.y + max.y) / 2};
        DrawRect(image,
                 center,
                 std::abs(max.x - min.x) / 2,
                 std::abs(max.y - min.y) / 2,
                 color);
        break;
    }
    case DrawCommandType::Frame:
        break;
    case DrawCommandType::ContactPoint:
        DrawDisk(image, position, 3, color);
        DrawLine(image,
                 position,
                 Project(command.position + command.end.Normalized() * 0.1f, options),
                 color);
        break;
    }
}

} // namespace

size_t DebugRasterImage::NonBackgroundPixelCount() const {
    size_t count = 0;
    for (const auto& pixel : pixels) {
        if (pixel.r != background.r || pixel.g != background.g ||
            pixel.b != background.b || pixel.a != background.a) {
            ++count;
        }
    }
    return count;
}

bool DebugRasterImage::WritePpm(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& pixel : pixels) {
        out.put(static_cast<char>(pixel.r));
        out.put(static_cast<char>(pixel.g));
        out.put(static_cast<char>(pixel.b));
    }
    return out.good();
}

DebugRasterImage RasterizeDebugDrawList(const DebugDrawList& list,
                                        const DebugRasterOptions& options) {
    DebugRasterImage image;
    image.width = options.width;
    image.height = options.height;
    image.background = options.background;
    image.pixels.assign(static_cast<size_t>(image.width) * image.height, image.background);

    for (const auto& command : list.Commands()) {
        RasterizeCommand(image, command, options);
    }

    return image;
}

} // namespace nuka::app
