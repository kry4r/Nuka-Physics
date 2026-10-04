
#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "render/texture_image.hpp"
#include "render/viewer_resources.hpp"
#include "viewer_ui_font.hpp"

#include <vulkan/vulkan.h>

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

#include <cstdio>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>

#ifndef NUKA_IMGUI_FONT_UI_TTF
#define NUKA_IMGUI_FONT_UI_TTF ""
#endif
#ifndef NUKA_IMGUI_FONT_HEADING_TTF
#define NUKA_IMGUI_FONT_HEADING_TTF ""
#endif

namespace nuka::render::imgui {

namespace {

using namespace theme;

ImVec4 WithAlpha(const ImVec4& c, float a) { return ImVec4(c.x, c.y, c.z, a); }

}  // namespace

void ApplyNukaTheme() {
    ImGuiStyle& style = ImGui::GetStyle();

    // ---- Geometry: modern rounded panels, generous-but-tidy spacing ----------
    style.WindowRounding    = 0.0f;   // rounded modern window corners
    style.ChildRounding     = 0.0f;
    style.PopupRounding     = 8.0f;
    style.FrameRounding     = 5.0f;   // inputs/sliders/buttons
    style.GrabRounding      = 5.0f;   // slider grab pills
    style.TabRounding       = 5.0f;
    style.ScrollbarRounding = 8.0f;

    style.WindowPadding     = ImVec2(14.0f, 12.0f);
    style.FramePadding      = ImVec2(8.0f, 7.0f);
    style.CellPadding       = ImVec2(8.0f, 5.0f);
    style.ItemSpacing       = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing  = ImVec2(7.0f, 6.0f);
    style.IndentSpacing     = 18.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 11.0f;

    // Subtle 1px borders; a hint of separation, not heavy chrome.
    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.TabBorderSize     = 0.0f;

    style.WindowTitleAlign  = ImVec2(0.02f, 0.5f);  // title nudged off the corner
    style.WindowMenuButtonPosition = ImGuiDir_None; // no collapse arrow clutter

    // Anti-aliasing on (curves look intentional, not jagged).
    style.AntiAliasedLines       = true;
    style.AntiAliasedLinesUseTex = true;
    style.AntiAliasedFill        = true;

    // ---- Colors: derive everything from the palette tokens above -------------
    ImVec4* c = style.Colors;

    c[ImGuiCol_Text]                 = kText;
    c[ImGuiCol_TextDisabled]         = kTextDim;

    c[ImGuiCol_WindowBg]             = kBgPanel;
    c[ImGuiCol_ChildBg]              = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg]              = kTopBar;

    c[ImGuiCol_Border]               = kLine;
    c[ImGuiCol_BorderShadow]         = ImVec4(0, 0, 0, 0);

    c[ImGuiCol_FrameBg]              = kBgRaised;
    c[ImGuiCol_FrameBgHovered]       = kBgHover;
    c[ImGuiCol_FrameBgActive]        = kBgActive;

    // Title bar: the void color, with the accent reserved for the ACTIVE window.
    c[ImGuiCol_TitleBg]              = kBgVoid;
    c[ImGuiCol_TitleBgActive]        = kBgRaised;
    c[ImGuiCol_TitleBgCollapsed]     = WithAlpha(kBgVoid, 0.80f);
    c[ImGuiCol_MenuBarBg]            = kTopBar;

    c[ImGuiCol_ScrollbarBg]          = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab]        = kBgActive;
    c[ImGuiCol_ScrollbarGrabHovered] = WithAlpha(kAccent, 0.55f);
    c[ImGuiCol_ScrollbarGrabActive]  = kAccent;

    // The accent shows up exactly where interaction lives.
    c[ImGuiCol_CheckMark]            = kAccent;
    c[ImGuiCol_SliderGrab]           = kAccent;
    c[ImGuiCol_SliderGrabActive]     = WithAlpha(kAccent, 1.00f);

    c[ImGuiCol_Button]               = kBgRaised;
    c[ImGuiCol_ButtonHovered]        = kBgHover;
    c[ImGuiCol_ButtonActive]         = kBgActive;

    c[ImGuiCol_Header]               = kBgActive;
    c[ImGuiCol_HeaderHovered]        = kBgHover;
    c[ImGuiCol_HeaderActive]         = kBgActive;

    c[ImGuiCol_Separator]            = kLine;
    c[ImGuiCol_SeparatorHovered]     = WithAlpha(kAccent, 0.55f);
    c[ImGuiCol_SeparatorActive]      = kAccent;

    c[ImGuiCol_ResizeGrip]           = WithAlpha(kAccent, 0.20f);
    c[ImGuiCol_ResizeGripHovered]    = WithAlpha(kAccent, 0.50f);
    c[ImGuiCol_ResizeGripActive]     = kAccent;

    c[ImGuiCol_Tab]                  = kBgRaised;
    c[ImGuiCol_TabHovered]           = WithAlpha(kAccent, 0.45f);
    c[ImGuiCol_TabSelected]          = kBgActive;
    c[ImGuiCol_TabSelectedOverline]  = kAccent;             // the accent underline on the active tab
    c[ImGuiCol_TabDimmed]            = kBgPanel;
    c[ImGuiCol_TabDimmedSelected]    = kBgRaised;

    c[ImGuiCol_PlotLines]            = kAccent;
    c[ImGuiCol_PlotLinesHovered]     = WithAlpha(kAccent, 1.00f);
    c[ImGuiCol_PlotHistogram]        = kAccentDim;
    c[ImGuiCol_PlotHistogramHovered] = kAccent;

    c[ImGuiCol_TableHeaderBg]        = kBgVoid;
    c[ImGuiCol_TableBorderStrong]    = kLine;
    c[ImGuiCol_TableBorderLight]     = WithAlpha(kLine, 0.55f);
    c[ImGuiCol_TableRowBg]           = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt]        = WithAlpha(kBgRaised, 0.35f);

    c[ImGuiCol_TextSelectedBg]       = WithAlpha(kAccent, 0.32f);
    c[ImGuiCol_DragDropTarget]       = kAccent;
    c[ImGuiCol_NavCursor]            = kAccent;
    c[ImGuiCol_NavWindowingHighlight]= WithAlpha(kAccent, 0.70f);
    c[ImGuiCol_NavWindowingDimBg]    = WithAlpha(kBgVoid, 0.55f);
    c[ImGuiCol_ModalWindowDimBg]     = WithAlpha(kBgVoid, 0.55f);

    // Docking (docking branch): the empty central node uses the void, the preview
    // overlay uses the accent so dock targets read clearly.
    c[ImGuiCol_DockingPreview]       = WithAlpha(kAccent, 0.45f);
    c[ImGuiCol_DockingEmptyBg]       = kBgVoid;
}

namespace {
struct FontRoles {
    ImFontAtlas* atlas = nullptr;
    ImFont* body = nullptr;
    ImFont* heading = nullptr;
    ImFont* mono = nullptr;
    ImFontAtlasRectId logo = ImFontAtlasRectId_Invalid;
    bool cjk = false;
    std::vector<unsigned char> cjk_data;
};
FontRoles fonts;

bool FontExists(const char* path) {
    std::error_code error;
    return path && path[0] && std::filesystem::is_regular_file(path, error);
}

void MergeCjk(float size) {
    if (fonts.cjk_data.empty()) {
        const auto path = nuka::render::ViewerResource("assets/viewer/fonts/NotoSansCJKsc-Regular.otf", NUKA_IMGUI_FONT_CJK);
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) return;
        const auto bytes = input.tellg();
        if (bytes <= 0 || bytes > 32 * 1024 * 1024) return;
        fonts.cjk_data.resize(static_cast<size_t>(bytes));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(fonts.cjk_data.data()), bytes)) { fonts.cjk_data.clear(); return; }
    }
    ImFontConfig cfg;
    cfg.MergeMode = true;
    cfg.FontDataOwnedByAtlas = false;
    fonts.cjk = ImGui::GetIO().Fonts->AddFontFromMemoryTTF(fonts.cjk_data.data(),
                 static_cast<int>(fonts.cjk_data.size()), size, &cfg) != nullptr;
}

}  // namespace

bool HasNukaCjkFont() { return fonts.atlas == ImGui::GetIO().Fonts && fonts.cjk; }

ImFont* GetNukaFont(FontRole role) {
    if (fonts.atlas != ImGui::GetIO().Fonts) return nullptr;
    if (role == FontRole::Heading) return fonts.heading;
    if (role == FontRole::Mono) return fonts.mono;
    return fonts.body;
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
    const auto body_path = nuka::render::ViewerResource("assets/viewer/fonts/Inter-Regular.ttf", NUKA_IMGUI_FONT_UI_TTF);
    const auto heading_path = nuka::render::ViewerResource("assets/viewer/fonts/Inter-SemiBold.ttf", NUKA_IMGUI_FONT_HEADING_TTF);
    const auto mono_path = nuka::render::ViewerResource("assets/viewer/fonts/JetBrainsMono-Regular.ttf", NUKA_IMGUI_FONT_MONO);
    if (FontExists(body_path.c_str()))
        fonts.body = io.Fonts->AddFontFromFileTTF(body_path.c_str(), 14.0f);
    if (!fonts.body) {
        ImFontConfig cfg;
        cfg.FontDataOwnedByAtlas = false;
        fonts.body = io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kEmbeddedUiFont),
                                                   static_cast<int>(sizeof(kEmbeddedUiFont)), 14.0f, &cfg);
    }
    MergeCjk(14.0f);
    io.FontDefault = fonts.body;
    if (FontExists(heading_path.c_str())) {
        fonts.heading = io.Fonts->AddFontFromFileTTF(heading_path.c_str(), 14.0f);
        MergeCjk(14.0f);
    } else fonts.heading = fonts.body;
    if (FontExists(mono_path.c_str())) {
        fonts.mono = io.Fonts->AddFontFromFileTTF(mono_path.c_str(), 13.0f);
        MergeCjk(13.0f);
    } else fonts.mono = fonts.body;
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
    if (!fonts.cjk) std::fprintf(stderr, "[nuka_imgui] CJK font missing: package assets/viewer/fonts/NotoSansCJKsc-Regular.otf; Chinese glyphs unavailable\n");
    return fonts.body != nullptr;
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

    ApplyNukaTheme();
    LoadNukaFonts();

    ImGui_ImplVulkan_InitInfo vk = ToVulkanInitInfo(info);
    if (!ImGui_ImplVulkan_Init(&vk)) {
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
    ImGui::DestroyContext();
    fonts = {};
    initialized_ = false;
}

}  // namespace nuka::render::imgui
