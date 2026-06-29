// ---------------------------------------------------------------------------
// .nks /script node roundtrip fidelity.
// ---------------------------------------------------------------------------
// A SceneIR carrying authored /script nodes (inline source + a stable node id +
// an explicit parent tree-path) is Saved to .nks and Loaded back; the loaded
// ScriptRecords must be field-equal AND the facade must re-project each as a
// first-class /script node bound to a ScriptComponent under the authored parent.
// A separate case proves a script-FREE scene (examples/scenes/go2.nks) re-saves
// byte-identical with no "scripts" key -- the additive, media-style looser gate:
// the section is emitted only when present, so the strict byte-identity the
// script-free fixtures rely on is untouched.
// ---------------------------------------------------------------------------

#include "scene/ecs/components.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nuka::scene;
namespace fs = std::filesystem;

namespace {

std::vector<uint8_t> ReadBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}
std::string ReadText(const fs::path& p) {
    const std::vector<uint8_t> b = ReadBytes(p);
    return std::string(b.begin(), b.end());
}

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/nks_script_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    fs::path SubFile(const std::string& subdir, const std::string& name) const {
        const fs::path d = path_ / subdir;
        std::error_code ec;
        fs::create_directories(d, ec);
        return d / name;
    }
private:
    fs::path path_;
};

// A scene with a rigid body (a parent for one script) + two /script nodes: one at
// root, one parented under the body. The source carries newlines, quotes and a
// non-ASCII byte so the JSON string escaping round-trip is exercised.
SceneIR BuildScriptScene() {
    SceneIR s;

    RigidBodyRecord body;
    body.name = "robot";
    body.is_static = true;
    s.AddRigidBody(std::move(body));

    ScriptRecord root_script;
    root_script.name = "spinner";
    root_script.stable_id = 7;
    root_script.source =
        "import nuka\n"
        "STEP = 0  # \"quoted\" + unicode \xc2\xb5\n"
        "def on_step():\n"
        "    global STEP\n"
        "    STEP += 1\n"
        "    nuka.set_drive(0, float(STEP))\n";
    s.AddScript(std::move(root_script));

    ScriptRecord child_script;
    child_script.name = "attached";
    child_script.stable_id = 42;
    child_script.parent_path = "robot";
    child_script.source = "def on_ready():\n    nuka.log('ready')\n";
    s.AddScript(std::move(child_script));

    return s;
}

void ExpectScriptEqual(const ScriptRecord& a, const ScriptRecord& b) {
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.stable_id, b.stable_id);
    EXPECT_EQ(a.parent_path, b.parent_path);
    EXPECT_EQ(a.source, b.source);
}

}  // namespace

// 1. /script nodes round-trip field-equal AND re-project to first-class nodes with
//    a ScriptComponent under the authored parent path.
TEST(NksScriptRoundtrip, ScriptNodesRoundtrip) {
    TempDir tmp;
    const SceneIR orig = BuildScriptScene();
    const fs::path nks = tmp.SubFile("run1", "scripted.nks");
    nks::Save(orig, nks.string());

    // The saved JSON carries the additive "scripts" array + the inline source.
    const std::string text = ReadText(nks);
    EXPECT_NE(text.find("\"scripts\""), std::string::npos)
        << "saved .nks lacks the scripts section";
    EXPECT_NE(text.find("\"node_id\""), std::string::npos);
    EXPECT_NE(text.find("on_step"), std::string::npos)
        << "the inline script source was not serialized";

    const SceneIR loaded = nks::Load(nks.string());

    // record field-equality
    ASSERT_EQ(orig.ScriptCount(), 2u);
    ASSERT_EQ(loaded.ScriptCount(), orig.ScriptCount());
    for (ScriptId i = 0; i < orig.ScriptCount(); ++i)
        ExpectScriptEqual(orig.GetScript(i), loaded.GetScript(i));

    // the facade re-projects each script as a /script node bound to a
    // ScriptComponent that carries the same source + stable id.
    for (ScriptId i = 0; i < loaded.ScriptCount(); ++i) {
        const EntityId e = loaded.EntityOfScript(i);
        ASSERT_NE(e, kInvalidEntity) << "script " << i << " produced no entity";
        const ScriptComponent* sc = loaded.Ecs().Get<ScriptComponent>(e);
        ASSERT_NE(sc, nullptr) << "script node " << i << " has no ScriptComponent";
        EXPECT_EQ(sc->source, loaded.GetScript(i).source);
        EXPECT_EQ(sc->stable_id, loaded.GetScript(i).stable_id);
    }

    // the parented script hangs under "robot" (the explicit parent path survives).
    const EntityId child = loaded.EntityOfScript(1);
    const auto node = loaded.Ecs().NodeOf(child);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(loaded.Tree().PathOf(node), "robot/attached")
        << "the script's authored parent path was not honored";

    // determinism: Save -> Load -> Save is byte-identical.
    const fs::path nks2 = tmp.SubFile("run2", "scripted.nks");
    nks::Save(loaded, nks2.string());
    EXPECT_EQ(ReadBytes(nks), ReadBytes(nks2))
        << "scripted .nks not byte-identical on re-save";
}

// 2. A script-FREE scene re-saves byte-identical with NO scripts key -- the
//    additive guarantee that keeps the strict byte-identity gates intact.
TEST(NksScriptRoundtrip, ScriptFreeSceneAdditiveDeterministic) {
    const fs::path go2 = fs::path("examples/scenes/go2.nks");
    if (!fs::exists(go2)) GTEST_SKIP() << "examples/scenes/go2.nks not available";

    const SceneIR loaded = nks::Load(go2.string());
    EXPECT_EQ(loaded.ScriptCount(), 0u) << "go2.nks must carry no scripts";

    TempDir tmp;
    const fs::path nks1 = tmp.SubFile("run1", "go2.nks");
    nks::Save(loaded, nks1.string());
    const SceneIR reloaded = nks::Load(nks1.string());
    const fs::path nks2 = tmp.SubFile("run2", "go2.nks");
    nks::Save(reloaded, nks2.string());

    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2)) << "go2.nks not byte-identical on re-save";
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "go2.nka")),
              ReadBytes(tmp.SubFile("run2", "go2.nka")))
        << "go2.nka not byte-identical on re-save";
    const std::string text1 = ReadText(nks1);
    EXPECT_EQ(text1.find("\"scripts\""), std::string::npos)
        << "a script-free scene must emit no scripts section";
}
