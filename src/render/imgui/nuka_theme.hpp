#pragma once
// The editor's design tokens: colors, sizes and Phosphor icon glyphs. Widgets take their
// colors and spacing from here and never hard-code them.

#include "imgui.h"

#include <cstdint>

namespace nuka::render::imgui::theme {

constexpr ImVec4 Hex(uint32_t rgb, float alpha = 1.0f) {
    return ImVec4(static_cast<float>((rgb >> 16) & 0xFFu) / 255.0f, static_cast<float>((rgb >> 8) & 0xFFu) / 255.0f,
                  static_cast<float>(rgb & 0xFFu) / 255.0f, alpha);
}
constexpr ImVec4 Alpha(const ImVec4& c, float alpha) { return ImVec4(c.x, c.y, c.z, alpha); }

inline constexpr ImVec4 kBg0 = Hex(0x0B0C0F);       // viewport frame, menu bar, void
inline constexpr ImVec4 kBg1 = Hex(0x111317);       // panels
inline constexpr ImVec4 kBg2 = Hex(0x191C21);       // inputs, hovered rows
inline constexpr ImVec4 kBg3 = Hex(0x23272E);       // pressed
inline constexpr ImVec4 kLine = Hex(0x2A2F37);      // hairlines and borders
inline constexpr ImVec4 kLineHi = Hex(0x3B424D);
inline constexpr ImVec4 kFg = Hex(0xE6E9EE);
inline constexpr ImVec4 kFgMuted = Hex(0x9AA3AE);
inline constexpr ImVec4 kFgDim = Hex(0x636C78);
inline constexpr ImVec4 kAccent = Hex(0xA3D64A);    // focus, selection, playing
inline constexpr ImVec4 kAccentDim = Hex(0x6E9A2C);
inline constexpr ImVec4 kAccentInk = Hex(0x0D1206); // text on an accent fill
inline constexpr ImVec4 kOk = Hex(0x5CC76A);
inline constexpr ImVec4 kWarn = Hex(0xF0B429);
inline constexpr ImVec4 kErr = Hex(0xF05D5E);
inline constexpr ImVec4 kInfo = Hex(0x56B6F0);
inline constexpr ImVec4 kAxisX = Hex(0xE5534B);
inline constexpr ImVec4 kAxisY = Hex(0x6CC24A);
inline constexpr ImVec4 kAxisZ = Hex(0x4D8FE6);

// Sizes in unscaled pixels on a 4 px grid; the UI scale multiplies all of them.
inline constexpr float kFontBody = 14.0f;
inline constexpr float kFontMono = 13.0f;
inline constexpr float kFontLabel = 11.0f;  // display face, upper case: section and HUD labels
inline constexpr float kFontHero = 20.0f;
inline constexpr float kFontIcon = 16.0f;
inline constexpr float kSpace = 4.0f;
inline constexpr float kGroupGap = 12.0f;
inline constexpr float kRadius = 3.0f;
inline constexpr float kRadiusOverlay = 6.0f;
inline constexpr float kChamfer = 6.0f;
inline constexpr float kTitleBarHeight = 32.0f;  // the menu bar, which is also the window's title bar
inline constexpr float kWindowButton = 46.0f;    // width of the minimize, maximize and close buttons
inline constexpr float kResizeEdge = 5.0f;       // the band along a restored window's edges that resizes it
inline constexpr float kToolbarHeight = 36.0f;
inline constexpr float kLabelColumn = 0.40f;  // share of a field row taken by its label

}  // namespace nuka::render::imgui::theme

// Phosphor Icons 2.1.1 (MIT). The regular face is merged into every text face; the fill face
// (FontRole::IconFill) draws the same codepoints for active states.
namespace nuka::render::imgui::icon {

inline constexpr char32_t kFirst = 0xE000;  // the codepoint range the icon face owns
inline constexpr char32_t kLast = 0xEE82;

inline constexpr const char kPlay[] = "\xEE\x8F\x90";
inline constexpr const char kPause[] = "\xEE\x8E\x9E";
inline constexpr const char kStep[] = "\xEE\x96\xA6";
inline constexpr const char kReset[] = "\xEE\x80\xB8";
inline constexpr const char kUndo[] = "\xEE\x82\x8A";
inline constexpr const char kRedo[] = "\xEE\x82\x8C";
inline constexpr const char kSpeed[] = "\xEE\x98\xA8";
inline constexpr const char kMove[] = "\xEE\x82\xA4";
inline constexpr const char kRotate[] = "\xEE\x80\xB6";
inline constexpr const char kWorldSpace[] = "\xEE\x8A\x8E";
inline constexpr const char kLocalSpace[] = "\xEE\xB4\x8A";
inline constexpr const char kSnap[] = "\xEE\x9A\x82";
inline constexpr const char kGizmo[] = "\xEE\x9F\x88";
inline constexpr const char kZeroCopy[] = "\xEE\x8B\x9E";
inline constexpr const char kHostCopy[] = "\xEE\x87\x8A";
inline constexpr const char kWarn[] = "\xEE\x93\xA0";
inline constexpr const char kError[] = "\xEE\x93\xB8";
inline constexpr const char kInfo[] = "\xEE\x8B\x8E";
inline constexpr const char kOk[] = "\xEE\x86\x84";
inline constexpr const char kDot[] = "\xEE\x86\x8A";
inline constexpr const char kOpen[] = "\xEE\x89\x96";
inline constexpr const char kSave[] = "\xEE\x89\x88";
inline constexpr const char kSaveAs[] = "\xEE\xAB\xB4";
inline constexpr const char kNew[] = "\xEE\x88\xB6";
inline constexpr const char kClose[] = "\xEE\x93\xB6";
inline constexpr const char kFile[] = "\xEE\x88\xB0";
inline constexpr const char kUnload[] = "\xEE\x88\x92";
inline constexpr const char kStage[] = "\xEE\x99\xBC";
inline constexpr const char kProperty[] = "\xEE\x90\xB4";
inline constexpr const char kStats[] = "\xEE\x85\x94";
inline constexpr const char kDebug[] = "\xEE\x97\xB4";
inline constexpr const char kCamera[] = "\xEE\x93\x9A";
inline constexpr const char kDrive[] = "\xEE\xA6\xAC";
inline constexpr const char kScript[] = "\xEE\x86\xBC";
inline constexpr const char kConsole[] = "\xEE\xAB\xA8";
inline constexpr const char kInspector[] = "\xEE\xB9\x88";
inline constexpr const char kSearch[] = "\xEE\x8C\x8C";
inline constexpr const char kSettings[] = "\xEE\x89\xB2";
inline constexpr const char kExtensions[] = "\xEE\x96\x96";
inline constexpr const char kLayout[] = "\xEE\x9B\x96";
inline constexpr const char kHelp[] = "\xEE\x8F\xA8";
inline constexpr const char kRobot[] = "\xEE\x9D\xA2";
inline constexpr const char kSoft[] = "\xEE\x95\xA6";
inline constexpr const char kCloth[] = "\xEE\x89\x84";
inline constexpr const char kFluid[] = "\xEE\x88\x90";
inline constexpr const char kBody[] = "\xEE\x87\x9A";
inline constexpr const char kStatic[] = "\xEE\x97\x98";
inline constexpr const char kJoint[] = "\xEE\x8B\xA6";
inline constexpr const char kCollider[] = "\xEE\x9B\x8E";
inline constexpr const char kVisual[] = "\xEE\x88\xA0";
inline constexpr const char kCameraNode[] = "\xEE\x84\x8E";
inline constexpr const char kLight[] = "\xEE\x8B\x9C";
inline constexpr const char kGroup[] = "\xEE\x91\xA6";
inline constexpr const char kScene[] = "\xEE\x99\x92";
inline constexpr const char kScriptNode[] = "\xEE\xA4\x94";
inline constexpr const char kChevronRight[] = "\xEE\x84\xBA";
inline constexpr const char kChevronDown[] = "\xEE\x84\xB6";
inline constexpr const char kAdd[] = "\xEE\x8F\x94";
inline constexpr const char kTrash[] = "\xEE\x92\xA6";
inline constexpr const char kEdit[] = "\xEE\x8E\xB4";
inline constexpr const char kLive[] = "\xEE\x8B\x9E";
inline constexpr const char kRebuild[] = "\xEE\x82\x94";
inline constexpr const char kRun[] = "\xEE\x8F\x92";
inline constexpr const char kStop[] = "\xEE\x91\xAC";
inline constexpr const char kRepeat[] = "\xEE\x8F\xB6";
inline constexpr const char kCursorText[] = "\xEE\x9F\x98";
inline constexpr const char kFunction[] = "\xEE\xAF\xA4";
inline constexpr const char kTerminal[] = "\xEE\x91\xBE";
inline constexpr const char kWarnCircle[] = "\xEE\x93\xA2";
inline constexpr const char kClear[] = "\xEE\x88\x9E";
inline constexpr const char kFrame[] = "\xEE\x98\xA6";
inline constexpr const char kHome[] = "\xEE\x8B\x82";
inline constexpr const char kTarget[] = "\xEE\x87\x96";
inline constexpr const char kBox[] = "\xEE\x87\x9A";
inline constexpr const char kSphere[] = "\xEE\xB9\xA6";
inline constexpr const char kCapsule[] = "\xEE\x9C\x80";
inline constexpr const char kPlane[] = "\xEE\x91\x9E";
inline constexpr const char kCylinder[] = "\xEE\xA3\xBC";
inline constexpr const char kChip[] = "\xEE\x98\x90";
inline constexpr const char kTimer[] = "\xEE\x92\x92";
inline constexpr const char kKeyboard[] = "\xEE\x8B\x98";
inline constexpr const char kHexagon[] = "\xEE\x8A\xAE";
inline constexpr const char kCommand[] = "\xEE\x87\x84";
inline constexpr const char kAtom[] = "\xEE\x97\xA4";
inline constexpr const char kPalette[] = "\xEE\x9B\x88";
inline constexpr const char kLock[] = "\xEE\x8C\x88";
inline constexpr const char kFilter[] = "\xEE\x89\xA8";
inline constexpr const char kPlot[] = "\xEE\x85\x96";
inline constexpr const char kWaves[] = "\xEE\x9B\x9E";
inline constexpr const char kTranslate[] = "\xEE\x92\xA2";
inline constexpr const char kPower[] = "\xEE\x8F\x9A";
inline constexpr const char kProhibit[] = "\xEE\x8F\x9E";
inline constexpr const char kPlug[] = "\xEE\xA5\x86";
inline constexpr const char kPackage[] = "\xEE\x8E\x90";
inline constexpr const char kCode[] = "\xEE\x86\xBC";
inline constexpr const char kFilePy[] = "\xEE\xAC\xAC";
inline constexpr const char kFileCpp[] = "\xEE\xAC\xAE";
inline constexpr const char kRecent[] = "\xEE\x86\xA0";
inline constexpr const char kExit[] = "\xEE\x90\xAA";
inline constexpr const char kNewWindow[] = "\xEE\x97\x9E";
inline constexpr const char kImport[] = "\xEE\x88\x8C";
inline constexpr const char kRecover[] = "\xEE\x98\xBA";
inline constexpr const char kTemplate[] = "\xEE\x91\xA4";
inline constexpr const char kExample[] = "\xEE\x9E\x9E";
inline constexpr const char kDocs[] = "\xEE\x83\xA6";
inline constexpr const char kSparkle[] = "\xEE\x9A\xA2";
inline constexpr const char kEyeSlash[] = "\xEE\x88\xA4";
inline constexpr const char kLockOpen[] = "\xEE\x8C\x8A";
inline constexpr const char kFolder[] = "\xEE\x89\x8A";
inline constexpr const char kFolderPlus[] = "\xEE\x89\x98";
inline constexpr const char kFolderMinus[] = "\xEE\x89\x94";
inline constexpr const char kShapes[] = "\xEE\xB1\x9E";
inline constexpr const char kMagicWand[] = "\xEE\x9A\xB6";
inline constexpr const char kCopy[] = "\xEE\x87\x8A";
inline constexpr const char kPaste[] = "\xEE\x86\x98";
inline constexpr const char kGrid[] = "\xEE\x91\xA4";
inline constexpr const char kHull[] = "\xEE\x9B\x90";
inline constexpr const char kMeshShape[] = "\xEE\xB1\xBC";
inline constexpr const char kTimeline[] = "\xEE\x9E\x92";   // film-strip
inline constexpr const char kFork[] = "\xEE\x89\xBE";       // git-fork
inline constexpr const char kBranch[] = "\xEE\x89\xB8";     // git-branch
inline constexpr const char kGhost[] = "\xEE\x98\xAA";
inline constexpr const char kSkipBack[] = "\xEE\x96\xA4";
inline constexpr const char kCaretLeft[] = "\xEE\x84\xB8";
inline constexpr const char kKeyframe[] = "\xEE\x87\xAC";   // diamond
inline constexpr const char kGrab[] = "\xEE\x95\xBC";       // hand-grabbing
inline constexpr const char kPin[] = "\xEE\x8F\xA2";        // push-pin
inline constexpr const char kFilm[] = "\xEE\x9E\x92";       // film-strip
inline constexpr const char kViewSingle[] = "\xEE\x91\x9E";  // square
inline constexpr const char kViewSplit[] = "\xEE\xA1\xB0";   // square-split-horizontal
inline constexpr const char kViewQuad[] = "\xEE\x91\xA4";    // squares-four
inline constexpr const char kMaximize[] = "\xEE\x87\x90";    // corners-out
inline constexpr const char kRuler[] = "\xEE\x9A\xB8";
inline constexpr const char kOverlays[] = "\xEE\xB1\xBC";    // cube-transparent
inline constexpr const char kGamepad[] = "\xEE\x89\xAE";     // game-controller

}  // namespace nuka::render::imgui::icon
