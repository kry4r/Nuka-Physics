// Dear ImGui glue: Vulkan backend and ImPlot lifetime, the token theme and the font atlas
// (Nerd Font UbuntuSans, MartianMono and GeistMono, each with Phosphor icons and Noto Sans CJK merged in).

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "render/texture_image.hpp"
#include "render/viewer_resources.hpp"

#include <vulkan/vulkan.h>

#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_vulkan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nuka::render::imgui {

void ApplyNukaTheme(float scale) {
    using namespace theme;
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();

    // Square panels and hairline borders; small radii only on controls and overlays.
    style.WindowRounding = 0.0f;
    style.ChildRounding = kRadius;
    style.PopupRounding = kRadiusOverlay;
    style.FrameRounding = kRadius;
    style.GrabRounding = kRadius;
    style.TabRounding = kRadius;
    style.ScrollbarRounding = kRadius;
    style.WindowPadding = ImVec2(2.0f * kSpace, 2.0f * kSpace);
    style.FramePadding = ImVec2(2.0f * kSpace, kSpace);
    style.CellPadding = ImVec2(2.0f * kSpace, 0.5f * kSpace);
    style.ItemSpacing = ImVec2(2.0f * kSpace, kSpace + 2.0f);
    style.ItemInnerSpacing = ImVec2(kSpace + 2.0f, kSpace);
    style.IndentSpacing = 4.0f * kSpace;
    style.ScrollbarSize = 2.0f * kSpace;
    style.GrabMinSize = 2.0f * kSpace;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.TabBorderSize = 0.0f;
    style.TabBarBorderSize = 1.0f;
    style.TabBarOverlineSize = 2.0f;
    style.DockingSeparatorSize = 2.0f;
    style.SeparatorTextBorderSize = 1.0f;
    style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.FontSizeBase = kFontBody;

    ImVec4* c = style.Colors;
    const ImVec4 none(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_Text] = kFg;
    c[ImGuiCol_TextDisabled] = kFgDim;
    c[ImGuiCol_WindowBg] = kBg1;
    c[ImGuiCol_ChildBg] = none;
    c[ImGuiCol_PopupBg] = Alpha(kBg0, 0.97f);
    c[ImGuiCol_Border] = kLine;
    c[ImGuiCol_BorderShadow] = none;
    c[ImGuiCol_FrameBg] = kBg2;
    c[ImGuiCol_FrameBgHovered] = kBg3;
    c[ImGuiCol_FrameBgActive] = Alpha(kAccent, 0.18f);
    c[ImGuiCol_TitleBg] = kBg0;
    c[ImGuiCol_TitleBgActive] = kBg0;
    c[ImGuiCol_TitleBgCollapsed] = kBg0;
    c[ImGuiCol_MenuBarBg] = kBg0;
    c[ImGuiCol_ScrollbarBg] = none;
    c[ImGuiCol_ScrollbarGrab] = kBg3;
    c[ImGuiCol_ScrollbarGrabHovered] = kLineHi;
    c[ImGuiCol_ScrollbarGrabActive] = kAccentDim;
    c[ImGuiCol_CheckMark] = kAccent;
    c[ImGuiCol_SliderGrab] = kAccentDim;
    c[ImGuiCol_SliderGrabActive] = kAccent;
    c[ImGuiCol_Button] = kBg2;
    c[ImGuiCol_ButtonHovered] = kBg3;
    c[ImGuiCol_ButtonActive] = Alpha(kAccent, 0.35f);
    c[ImGuiCol_Header] = Alpha(kAccent, 0.14f);
    c[ImGuiCol_HeaderHovered] = Alpha(kAccent, 0.22f);
    c[ImGuiCol_HeaderActive] = Alpha(kAccent, 0.30f);
    c[ImGuiCol_Separator] = kLine;
    c[ImGuiCol_SeparatorHovered] = Alpha(kAccent, 0.60f);
    c[ImGuiCol_SeparatorActive] = kAccent;
    c[ImGuiCol_ResizeGrip] = none;
    c[ImGuiCol_ResizeGripHovered] = Alpha(kAccent, 0.50f);
    c[ImGuiCol_ResizeGripActive] = kAccent;
    c[ImGuiCol_InputTextCursor] = kAccent;
    c[ImGuiCol_Tab] = kBg0;
    c[ImGuiCol_TabHovered] = kBg2;
    c[ImGuiCol_TabSelected] = kBg1;
    c[ImGuiCol_TabSelectedOverline] = kAccent;
    c[ImGuiCol_TabDimmed] = kBg0;
    c[ImGuiCol_TabDimmedSelected] = kBg1;
    c[ImGuiCol_TabDimmedSelectedOverline] = Alpha(kAccent, 0.35f);
    c[ImGuiCol_DockingPreview] = Alpha(kAccent, 0.35f);
    c[ImGuiCol_DockingEmptyBg] = kBg0;
    c[ImGuiCol_PlotLines] = kAccent;
    c[ImGuiCol_PlotLinesHovered] = kFg;
    c[ImGuiCol_PlotHistogram] = kAccentDim;
    c[ImGuiCol_PlotHistogramHovered] = kAccent;
    c[ImGuiCol_TableHeaderBg] = kBg0;
    c[ImGuiCol_TableBorderStrong] = kLine;
    c[ImGuiCol_TableBorderLight] = Alpha(kLine, 0.6f);
    c[ImGuiCol_TableRowBg] = none;
    c[ImGuiCol_TableRowBgAlt] = Alpha(kBg2, 0.5f);
    c[ImGuiCol_TextLink] = kInfo;
    c[ImGuiCol_TextSelectedBg] = Alpha(kAccent, 0.28f);
    c[ImGuiCol_TreeLines] = kLine;
    c[ImGuiCol_DragDropTarget] = kAccent;
    c[ImGuiCol_DragDropTargetBg] = Alpha(kAccent, 0.08f);
    c[ImGuiCol_UnsavedMarker] = kWarn;
    c[ImGuiCol_NavCursor] = kAccent;
    c[ImGuiCol_NavWindowingHighlight] = Alpha(kAccent, 0.70f);
    c[ImGuiCol_NavWindowingDimBg] = Alpha(kBg0, 0.60f);
    c[ImGuiCol_ModalWindowDimBg] = Alpha(kBg0, 0.60f);

    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;

    if (ImPlot::GetCurrentContext() == nullptr) return;
    ImPlotStyle& plot = ImPlot::GetStyle();
    plot = ImPlotStyle();
    plot.PlotPadding = ImVec2(2.0f * kSpace * scale, 2.0f * kSpace * scale);
    plot.LabelPadding = ImVec2(kSpace * scale, kSpace * scale);
    plot.LegendPadding = ImVec2(2.0f * kSpace * scale, 2.0f * kSpace * scale);
    plot.MinorAlpha = 0.5f;
    ImVec4* p = plot.Colors;
    p[ImPlotCol_FrameBg] = none;
    p[ImPlotCol_PlotBg] = kBg0;
    p[ImPlotCol_PlotBorder] = kLine;
    p[ImPlotCol_LegendBg] = Alpha(kBg0, 0.90f);
    p[ImPlotCol_LegendBorder] = kLine;
    p[ImPlotCol_LegendText] = kFgMuted;
    p[ImPlotCol_TitleText] = kFg;
    p[ImPlotCol_InlayText] = kFgMuted;
    p[ImPlotCol_AxisText] = kFgDim;
    p[ImPlotCol_AxisGrid] = Alpha(kLine, 0.80f);
    p[ImPlotCol_AxisTick] = kLine;
    p[ImPlotCol_Selection] = kAccent;
    p[ImPlotCol_Crosshairs] = kFgMuted;
}

namespace {

constexpr size_t kFaceCount = 4u;
constexpr std::array<const char*, kFaceCount> kFaceFiles = {
    "UbuntuSansNerdFontPropo-Regular.ttf", "UbuntuSansNerdFontPropo-Medium.ttf",
    "MartianMonoNerdFontPropo-Medium.ttf", "GeistMonoNerdFontPropo-Regular.otf"};
constexpr std::array<const char*, kFaceCount> kFaceBuildPaths = {
    NUKA_IMGUI_FONT_BODY, NUKA_IMGUI_FONT_STRONG, NUKA_IMGUI_FONT_DISPLAY, NUKA_IMGUI_FONT_MONO};
constexpr std::array<float, kFaceCount> kFaceSizes = {theme::kFontBody, theme::kFontBody, theme::kFontLabel,
                                                      theme::kFontMono};
// The display face's caps stand tall for their size, so its CJK is merged a size up to match.
constexpr std::array<float, kFaceCount> kCjkScale = {1.0f, 1.0f, 1.25f, 1.0f};
// The Nerd faces give up their private-use glyphs so the merged Phosphor icons win.
constexpr ImWchar kPrivateUse[] = {0xE000, 0xF8FF, 0};
constexpr float kIconScale = theme::kFontIcon / theme::kFontBody;

struct FontRoles {
    ImFontAtlas* atlas = nullptr;
    std::array<ImFont*, kFaceCount + 1u> faces{};
    ImFontAtlasRectId logo = ImFontAtlasRectId_Invalid;
    bool cjk = false;
    std::vector<unsigned char> cjk_data;
    std::vector<unsigned char> icon_data;
    std::vector<unsigned char> icon_fill_data;
};
FontRoles fonts;

bool FontExists(const std::string& path) {
    std::error_code error;
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

void ReadBlob(const char* relative, const char* build_path, std::vector<unsigned char>& out) {
    std::ifstream input(nuka::render::ViewerResource(relative, build_path), std::ios::binary | std::ios::ate);
    if (!input) return;
    const auto bytes = input.tellg();
    out.resize(static_cast<size_t>(bytes));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(out.data()), bytes);
}

// Every face shares one copy of each blob; the atlas only borrows it.
ImFont* AddBlob(std::vector<unsigned char>& data, float size, ImFontConfig cfg) {
    if (data.empty()) return nullptr;
    cfg.FontDataOwnedByAtlas = false;
    return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(data.data(), static_cast<int>(data.size()), size, &cfg);
}

ImFontConfig IconConfig(float size, bool merge) {
    ImFontConfig cfg;
    cfg.MergeMode = merge;
    cfg.GlyphMinAdvanceX = size;
    cfg.GlyphOffset = ImVec2(0.0f, merge ? 0.16f * size : 0.0f);
    return cfg;
}

}  // namespace

bool HasNukaCjkFont() { return fonts.atlas == ImGui::GetIO().Fonts && fonts.cjk; }

ImFont* GetNukaFont(FontRole role) {
    if (fonts.atlas != ImGui::GetIO().Fonts) return nullptr;
    return fonts.faces[static_cast<size_t>(role)];
}

bool DrawNukaLogo(float size) {
    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    ImFontAtlasRect rect;
    if (fonts.atlas != atlas || fonts.logo == ImFontAtlasRectId_Invalid || !atlas->GetCustomRect(fonts.logo, &rect)) return false;
    ImGui::Image(atlas->TexRef, ImVec2(size, size), rect.uv0, rect.uv1);
    return true;
}

bool LoadNukaFonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    fonts = {};
    fonts.atlas = io.Fonts;
    ReadBlob("assets/viewer/fonts/NotoSansCJKsc-Regular.otf", NUKA_IMGUI_FONT_CJK, fonts.cjk_data);
    ReadBlob("assets/viewer/fonts/Phosphor.ttf", NUKA_IMGUI_FONT_ICON, fonts.icon_data);
    ReadBlob("assets/viewer/fonts/Phosphor-Fill.ttf", NUKA_IMGUI_FONT_ICON_FILL, fonts.icon_fill_data);
    for (size_t i = 0; i < kFaceCount; ++i) {
        const std::string relative = std::string("assets/viewer/fonts/") + kFaceFiles[i];
        const std::string path = nuka::render::ViewerResource(relative.c_str(), kFaceBuildPaths[i]);
        if (!FontExists(path)) continue;
        ImFontConfig text;
        text.GlyphExcludeRanges = kPrivateUse;
        fonts.faces[i] = io.Fonts->AddFontFromFileTTF(path.c_str(), kFaceSizes[i], &text);
        const float icon_size = kFaceSizes[i] * kIconScale;
        AddBlob(fonts.icon_data, icon_size, IconConfig(icon_size, true));
        ImFontConfig cjk;
        cjk.MergeMode = true;
        cjk.GlyphOffset = ImVec2(0.0f, (kCjkScale[i] - 1.0f) * 0.5f * kFaceSizes[i]);
        fonts.cjk = AddBlob(fonts.cjk_data, kFaceSizes[i] * kCjkScale[i], cjk) != nullptr;
    }
    const bool loaded = fonts.faces[0] != nullptr;
    if (!loaded) fonts.faces[0] = io.Fonts->AddFontDefault();
    fonts.faces[kFaceCount] = AddBlob(fonts.icon_fill_data, theme::kFontIcon, IconConfig(theme::kFontIcon, false));
    for (ImFont*& face : fonts.faces)
        if (face == nullptr) face = fonts.faces[0];
    io.FontDefault = fonts.faces[0];
    const auto logo = nuka::render::LoadTexture(nuka::render::ViewerResource("docs/media/nuka-logo.png", NUKA_IMGUI_LOGO), false);
    if (logo.width > 0u && logo.height > 0u && logo.channels == 4u) {
        ImFontAtlasRect rect;
        io.Fonts->TexPixelsUseColors = true;
        fonts.logo = io.Fonts->AddCustomRect(static_cast<int>(logo.width), static_cast<int>(logo.height), &rect);
        if (fonts.logo != ImFontAtlasRectId_Invalid) {
            for (uint32_t y = 0u; y < logo.height; ++y) {
                auto* dst = static_cast<unsigned char*>(io.Fonts->TexData->GetPixelsAt(rect.x, rect.y + static_cast<int>(y)));
                for (uint32_t x = 0u; x < logo.width * 4u; ++x)
                    dst[x] = static_cast<unsigned char>(std::clamp(logo.texels[static_cast<size_t>(y) * logo.width * 4u + x] * 255.0f + 0.5f, 0.0f, 255.0f));
            }
        }
    }
    if (!loaded) std::fprintf(stderr, "[nuka_imgui] UI font missing: package assets/viewer/fonts/%s\n", kFaceFiles[0]);
    if (fonts.icon_data.empty()) std::fprintf(stderr, "[nuka_imgui] icon font missing: package assets/viewer/fonts/Phosphor.ttf\n");
    if (!fonts.cjk) std::fprintf(stderr, "[nuka_imgui] CJK font missing: package assets/viewer/fonts/NotoSansCJKsc-Regular.otf; Chinese glyphs unavailable\n");
    return loaded;
}

NukaImGuiContext::~NukaImGuiContext() { Shutdown(); }

NukaImGuiContext::NukaImGuiContext(NukaImGuiContext&& other) noexcept
    : initialized_(other.initialized_) {
    other.initialized_ = false;
}

NukaImGuiContext& NukaImGuiContext::operator=(NukaImGuiContext&& other) noexcept {
    if (this != &other) {
        Shutdown();
        initialized_       = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

namespace {

// Map the flat nuka init bundle into the upstream nested InitInfo. RenderPass /
// Subpass / MSAA live in PipelineInfoMain since imgui 1.92's 2025/09/26 change.
ImGui_ImplVulkan_InitInfo ToVulkanInitInfo(const NukaImGuiInitInfo& info) {
    ImGui_ImplVulkan_InitInfo vk{};
    vk.ApiVersion     = info.api_version != 0 ? info.api_version : VK_API_VERSION_1_0;
    vk.Instance       = reinterpret_cast<VkInstance>(info.instance);
    vk.PhysicalDevice = reinterpret_cast<VkPhysicalDevice>(info.physical_device);
    vk.Device         = reinterpret_cast<VkDevice>(info.device);
    vk.QueueFamily    = info.queue_family;
    vk.Queue          = reinterpret_cast<VkQueue>(info.queue);
    vk.DescriptorPool = reinterpret_cast<VkDescriptorPool>(info.descriptor_pool);
    vk.MinImageCount  = info.min_image_count;
    vk.ImageCount     = info.image_count;

    vk.PipelineInfoMain.RenderPass  = reinterpret_cast<VkRenderPass>(info.render_pass);
    vk.PipelineInfoMain.Subpass     = info.subpass;
    const uint32_t msaa = info.msaa_samples != 0
                              ? info.msaa_samples
                              : static_cast<uint32_t>(VK_SAMPLE_COUNT_1_BIT);
    vk.PipelineInfoMain.MSAASamples = static_cast<VkSampleCountFlagBits>(msaa);
    return vk;
}

}  // namespace

bool NukaImGuiContext::Init(const NukaImGuiInitInfo& info) {
    if (initialized_) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ApplyNukaTheme();
    LoadNukaFonts();

    ImGui_ImplVulkan_InitInfo vk = ToVulkanInitInfo(info);
    if (!ImGui_ImplVulkan_Init(&vk)) {
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        fonts = {};
        return false;
    }

    initialized_ = true;
    return true;
}

void NukaImGuiContext::NotifySwapchainRecreated(uint32_t min_image_count) {
    if (!initialized_) return;
    // The recreated present pass is render-pass-compatible, so the pipeline is reused;
    // only the image count needs a refresh (no-op when minImageCount is unchanged).
    ImGui_ImplVulkan_SetMinImageCount(min_image_count);
}

void NukaImGuiContext::NewFrame() {
    if (!initialized_) return;
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
}

void NukaImGuiContext::RenderDrawData(NukaVkCommandBuffer command_buffer) {
    if (!initialized_) return;
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data == nullptr) return;
    ImGui_ImplVulkan_RenderDrawData(draw_data,
                                    reinterpret_cast<VkCommandBuffer>(command_buffer));
}

void NukaImGuiContext::Shutdown() {
    if (!initialized_) return;
    ImGui_ImplVulkan_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    fonts = {};
    initialized_ = false;
}

}  // namespace nuka::render::imgui
