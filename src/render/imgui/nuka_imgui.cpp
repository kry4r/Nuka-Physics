// ---------------------------------------------------------------------------
// nuka::render::imgui -- glue implementation (M8.5 T1).
//
// Wraps the vendored Dear ImGui (external/imgui, v1.92.8-docking) + its Vulkan
// backend behind the thin nuka API in nuka_imgui.hpp, and defines the custom
// "Nuka" theme + the JetBrains Mono font load.
//
// The vendored ImGui headers/impl are SYSTEM includes (set in src/CMakeLists.txt)
// so their warnings are suppressed and they stay out of the zero-CUDA lint scope.
// This TU itself is plain host C++/Vulkan -- no CUDA.
// ---------------------------------------------------------------------------

#include "render/imgui/nuka_imgui.hpp"

#include <vulkan/vulkan.h>

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

#include <cstdio>

// Font paths are injected by CMake (target_compile_definitions), mirroring how the
// raster renderer receives its compiled-SPV paths. If they are somehow absent we
// fall back to the built-in default font.
#ifndef NUKA_IMGUI_FONT_UI_TTF
#define NUKA_IMGUI_FONT_UI_TTF ""
#endif
#ifndef NUKA_IMGUI_FONT_HEADING_TTF
#define NUKA_IMGUI_FONT_HEADING_TTF ""
#endif

namespace nuka::render::imgui {

namespace {

// ---------------------------------------------------------------------------
// THE NUKA PALETTE (documented for T3 so its panels stay consistent).
//
// Aesthetic target: a polished engineering / physics-sim PRO TOOL -- a deep,
// slightly blue-tinted charcoal base (not pure black, not gray) with ONE warm-cool
// accent: a bright TEAL/CYAN ("nuka teal"). Think a precision instrument: dark,
// calm, high-contrast text, with the accent reserved for interactive/active state
// (sliders, selected headers, active buttons, the resize grip). No second accent.
//
// All values are linear-ish sRGB 0..1 (ImGui stores colors that way). Names are
// the design tokens; T3 should reuse these rather than inventing new colors.
//
//   Base (darkest -> lightest):
//     bg_void     #0E1116  the window/app void (deepest)
//     bg_panel    #151A21  panel / child / popup background
//     bg_raised   #1C232C  frames (inputs, sliders, combo) -- one step raised
//     bg_hover    #232C37  hovered frame
//     bg_active   #2B3645  active/pressed frame
//     line        #2A323D  subtle borders / separators
//   Text:
//     text        #C6D0DA  primary text (soft off-white, easy on the eyes)
//     text_dim    #6E7A88  disabled / secondary text
//   Accent (the single teal):
//     accent      #2DD4BF  nuka teal -- active/selected/grab
//     accent_dim  #1E9C8C  teal pressed / darker
//     accent_soft #2DD4BF @ ~0.30a  teal wash (header bg, selected row)
// ---------------------------------------------------------------------------
constexpr ImVec4 kBgVoid    = ImVec4(0.055f, 0.067f, 0.086f, 1.00f);  // #0E1116
constexpr ImVec4 kBgPanel   = ImVec4(0.082f, 0.102f, 0.129f, 1.00f);  // #151A21
constexpr ImVec4 kBgRaised  = ImVec4(0.110f, 0.137f, 0.173f, 1.00f);  // #1C232C
constexpr ImVec4 kBgHover   = ImVec4(0.137f, 0.173f, 0.216f, 1.00f);  // #232C37
constexpr ImVec4 kBgActive  = ImVec4(0.169f, 0.212f, 0.271f, 1.00f);  // #2B3645
constexpr ImVec4 kLine      = ImVec4(0.165f, 0.196f, 0.239f, 1.00f);  // #2A323D
constexpr ImVec4 kText      = ImVec4(0.776f, 0.816f, 0.855f, 1.00f);  // #C6D0DA
constexpr ImVec4 kTextDim   = ImVec4(0.431f, 0.478f, 0.533f, 1.00f);  // #6E7A88
constexpr ImVec4 kAccent    = ImVec4(0.176f, 0.831f, 0.749f, 1.00f);  // #2DD4BF
constexpr ImVec4 kAccentDim = ImVec4(0.118f, 0.612f, 0.549f, 1.00f);  // #1E9C8C

ImVec4 WithAlpha(const ImVec4& c, float a) { return ImVec4(c.x, c.y, c.z, a); }

}  // namespace

void ApplyNukaTheme() {
    ImGuiStyle& style = ImGui::GetStyle();

    // ---- Geometry: modern rounded panels, generous-but-tidy spacing ----------
    style.WindowRounding    = 8.0f;   // rounded modern window corners
    style.ChildRounding     = 6.0f;
    style.PopupRounding     = 6.0f;
    style.FrameRounding     = 5.0f;   // inputs/sliders/buttons
    style.GrabRounding      = 5.0f;   // slider grab pills
    style.TabRounding       = 5.0f;
    style.ScrollbarRounding = 8.0f;

    style.WindowPadding     = ImVec2(12.0f, 12.0f);
    style.FramePadding      = ImVec2(10.0f, 6.0f);
    style.CellPadding       = ImVec2(8.0f, 5.0f);
    style.ItemSpacing       = ImVec2(10.0f, 8.0f);
    style.ItemInnerSpacing  = ImVec2(7.0f, 6.0f);
    style.IndentSpacing     = 18.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 11.0f;

    // Subtle 1px borders; a hint of separation, not heavy chrome.
    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
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

    // Window is SEMI-TRANSPARENT (0.96) -- a hint of the viewport beneath, the
    // "instrument glass" feel, without hurting readability.
    c[ImGuiCol_WindowBg]             = WithAlpha(kBgPanel, 0.96f);
    c[ImGuiCol_ChildBg]              = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg]              = WithAlpha(kBgVoid, 0.98f);

    c[ImGuiCol_Border]               = kLine;
    c[ImGuiCol_BorderShadow]         = ImVec4(0, 0, 0, 0);

    c[ImGuiCol_FrameBg]              = kBgRaised;
    c[ImGuiCol_FrameBgHovered]       = kBgHover;
    c[ImGuiCol_FrameBgActive]        = kBgActive;

    // Title bar: the void color, with the accent reserved for the ACTIVE window.
    c[ImGuiCol_TitleBg]              = kBgVoid;
    c[ImGuiCol_TitleBgActive]        = kBgRaised;
    c[ImGuiCol_TitleBgCollapsed]     = WithAlpha(kBgVoid, 0.80f);
    c[ImGuiCol_MenuBarBg]            = kBgVoid;

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
    c[ImGuiCol_ButtonActive]         = WithAlpha(kAccent, 0.85f);

    // Headers (collapsing/selectable/tree): a teal WASH so selection reads clearly.
    c[ImGuiCol_Header]               = WithAlpha(kAccent, 0.18f);
    c[ImGuiCol_HeaderHovered]        = WithAlpha(kAccent, 0.30f);
    c[ImGuiCol_HeaderActive]         = WithAlpha(kAccent, 0.42f);

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

bool LoadNukaFonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    // Sizes: a UI body size + a larger heading size (panel titles / hero stats).
    constexpr float kUiSize      = 16.0f;
    constexpr float kHeadingSize = 21.0f;

    const char* ui_ttf      = NUKA_IMGUI_FONT_UI_TTF;
    const char* heading_ttf = NUKA_IMGUI_FONT_HEADING_TTF;

    bool ui_loaded = false;
    if (ui_ttf && ui_ttf[0] != '\0') {
        // The default (first-added) font is the UI body font: JetBrains Mono.
        if (io.Fonts->AddFontFromFileTTF(ui_ttf, kUiSize) != nullptr) {
            ui_loaded = true;
        }
    }
    if (heading_ttf && heading_ttf[0] != '\0') {
        // Heading font (JetBrains Mono Bold); T3 pushes it for titles via index 1.
        io.Fonts->AddFontFromFileTTF(heading_ttf, kHeadingSize);
    }

    if (!ui_loaded) {
        // Vendored TTF unreadable -> fall back so the UI still renders. The custom
        // FONT requirement is then unmet (flagged here); the THEME still applies.
        std::fprintf(stderr,
            "[nuka_imgui] WARNING: vendored JetBrains Mono not loaded "
            "(ui_ttf='%s'); falling back to ImGui default font.\n",
            ui_ttf ? ui_ttf : "(null)");
        io.Fonts->AddFontDefault();
    }
    return ui_loaded;
}

// ---------------------------------------------------------------------------
// NukaImGuiContext
// ---------------------------------------------------------------------------
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

bool NukaImGuiContext::Init(const NukaImGuiInitInfo& info) {
    if (initialized_) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ApplyNukaTheme();
    LoadNukaFonts();

    // Map the flat nuka init bundle into the upstream nested InitInfo. RenderPass /
    // Subpass / MSAA live in PipelineInfoMain since imgui 1.92's 2025/09/26 change.
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

    if (!ImGui_ImplVulkan_Init(&vk)) {
        ImGui::DestroyContext();
        return false;
    }

    initialized_ = true;
    return true;
}

void NukaImGuiContext::NewFrame() {
    if (!initialized_) return;
    // NOTE (seam): the platform (xcb) backend -- T2 -- must already have set
    // io.DisplaySize / io.DeltaTime / mouse+keyboard for this frame. This glue only
    // drives the Vulkan backend half + ImGui::NewFrame.
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
    initialized_ = false;
}

}  // namespace nuka::render::imgui
