#pragma once
// The robot workflow's joint state as the physics thread samples it and the Joints panel shows it.

#include "math/transform.hpp"

#include <cstdint>
#include <cstddef>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

// The shown environment's joints at one step, one value per link.
struct JointSample {
    uint64_t tick = 0u;
    double time = 0.0;
    std::vector<float> q;
    std::vector<float> target;
    std::vector<float> effort;
};

// What a mirror publish reads of the shown environment's joints, one value per link.
struct JointFrame {
    std::vector<float> q;
    std::vector<float> qd;
    std::vector<float> target;
    std::vector<float> effort;
    std::vector<float> saturated;  // 1 where the actuator clamped its effort
};

// A moving joint: its link slot, its authored record and the drive on it.
struct RobotJoint {
    uint32_t link = 0u;
    std::string name;
    std::string path;  // the child link's node
    bool prismatic = false;
    bool limited = false;
    float lower = 0.0f;
    float upper = 0.0f;
    std::string drive;         // the actuator type: position, motor, velocity or force; empty without one
    float kp = 0.0f;           // a position drive's gain
    float force_limit = 0.0f;  // 0 is unlimited
    float damping = 0.0f;      // the joint's own
    float initial = 0.0f;      // the authored initial position, where Reset puts the target
};

// A link of the shown environment's articulations.
struct RobotLink {
    std::string path;
    int parent = -1;    // the parent link, by index into the links; -1 at a root
    std::string joint;  // the incoming joint, empty at a root
    std::string type;   // the incoming joint's type, "root" at a root
    float mass = 0.0f;
    math::Transform inertial;    // the principal inertia frame in the link's frame
    math::Vec3 inertia;          // the principal moments [kg·m²]
    math::Transform frame;       // the incoming joint's frame in the link's frame
    math::Transform zero_frame;  // that frame at q = 0, in the parent link's frame
    math::Vec3 axis{0.0f, 0.0f, 1.0f};  // the joint's axis in its frame
    float size = 0.0f;           // the longest edge of the link's visual bounds [m]
};

// Named joint positions in rad or m; applying one moves only the joints it names.
struct JointPose {
    std::string name;
    std::vector<std::pair<std::string, float>> joints;
};

// A gain test of one joint: sample 0 is the state it started from, sample i the state after
// step i, which drove `target[i]`.
struct GainTrace {
    uint64_t request = 0u;  // the test command's seq
    uint32_t link = 0u;
    bool sine = false;
    float amplitude = 0.0f;  // rad or m
    float frequency = 0.0f;  // Hz
    float base = 0.0f;       // the target before the test
    std::vector<float> time, target, q, qd, effort, saturated;
};

// What a gain test measured; NaN where a figure does not apply.
struct GainMetrics {
    static constexpr float kNone = std::numeric_limits<float>::quiet_NaN();
    float rise = kNone;          // s, from 10% to 90% of the settled change
    float overshoot = kNone;     // % of the settled change
    float peak_time = kNone;     // s
    float settling = kNone;      // s, after which it stays within 2% of the settled value
    float steady_error = kNone;  // the target minus the settled value
    float zeta = kNone;          // the damping ratio and natural frequency [rad/s] of a
    float omega = kNone;         // second-order fit to the overshoot and peak time
    float gain_db = kNone;       // sine: the response's amplitude over the command's
    float lag = kNone;           // sine: degrees behind the command
    float rmse = kNone;          // sine: tracking error over the second half
    float saturation = 0.0f;     // the share of steps at the force limit
};

// One Gain Tuner run: the joint, the gains it ran with, its trace and what it measured.
struct GainRun {
    std::string joint;
    bool prismatic = false;
    float kp = 0.0f;
    float damping = 0.0f;
    float force_limit = 0.0f;
    float duration = 0.0f;
    GainTrace trace;
    GainMetrics metrics;
};

inline constexpr size_t kGainRuns = 6u;  // the runs the tuner keeps

struct GainTuner {
    bool open = false;
    std::string joint;  // the joint a test drives
    bool sine = false;
    float amplitude = 0.1f;  // rad or m
    float frequency = 1.0f;  // Hz
    float duration = 1.0f;   // s
    uint64_t running = 0u;   // the running test's request; 0 while none runs
    GainRun next;            // its joint and gains
    std::deque<GainRun> runs;  // the newest last
};

// A quantity the System Identification window searches: a registry parameter on one joint or all of
// the demonstrations' joints, between `min` and `max`.
struct SysIdParam {
    std::string param = "joint_damping";  // joint_damping, joint_armature, joint_frictionloss or link_mass_scale
    std::string joint;                    // empty is every joint
    float min = 0.0f;
    float max = 1.0f;
};

// What a search settled on for one quantity: the joints' mean before it (1 for a scale) and after.
struct SysIdFound {
    std::string param;
    std::vector<std::string> joints;
    double nominal = 0.0;
    double value = 0.0;
};

// The System Identification window's form, the search's progress and what it found.
struct SysIdView {
    bool open = false;
    int episode = -1;  // by index into the demonstrations; -1 is the newest
    std::vector<SysIdParam> params = {SysIdParam{}};
    int generations = 20;
    int population = 8;
    bool running = false;
    uint32_t generation = 0u;
    uint32_t total = 0u;  // the search's generations
    uint32_t evaluations = 0u;
    double start_loss = 0.0;  // with the values in force [rad² or m²]
    double best_loss = 0.0;
    std::vector<double> curve;  // the best loss after each generation
    std::vector<SysIdFound> found;
    std::string error;
};

// The Robot Assembler's choices: an asset's link or a scene body, the link it hangs off and the joint.
struct Assembler {
    bool open = false;
    bool from_asset = true;
    std::string path;  // the asset
    std::string profile = "strict";
    std::vector<std::string> asset_links;  // the asset's links, its roots first
    std::string asset_error;
    std::string child;   // the scene body
    std::string parent;  // the link it hangs off
    std::string mount;   // the attached link the joint frame places; empty is the root
    int joint = 0;       // fixed, revolute or prismatic
    float axis[3] = {0.0f, 0.0f, 1.0f};
    bool keep_pose = true;  // a scene body stays where it is, else the mount goes to `pos`/`rot` in the parent
    float pos[3] = {0.0f, 0.0f, 0.0f};
    float rot[3] = {0.0f, 0.0f, 0.0f};  // XYZ degrees, as the inspector shows rotations
    std::string selected;  // the selected node, for Use Selection
    bool posed = false;    // the scene body's offset from the parent below is known
    float offset_pos[3] = {0.0f, 0.0f, 0.0f};
    float offset_rot[3] = {0.0f, 0.0f, 0.0f};
};

// How the gizmo drags a link through IK, and what the newest solve reached.
struct IkState {
    bool on = true;
    std::string root;               // the chain's root link; empty is the articulation root
    std::vector<std::string> pins;  // links a drag holds where they were when it began
    bool lock_orientation = true;   // a move keeps the link's orientation from the press
    float rotation_weight = 0.1f;   // m per rad
    uint64_t pending = 0u;          // the gizmo's solve in flight; 0 while none
    int link = -1;                  // the dragged link, by index into the links; -1 when none
    bool missed = false;            // the newest solve fell short of the dragged link's goal
    float error_mm = 0.0f;
    float error_deg = 0.0f;
    float target[3] = {0.0f, 0.0f, 0.0f};  // the dragged link's goal
    float at[3] = {0.0f, 0.0f, 0.0f};      // where the link is
    std::string error;
};

// The robot overlays the viewport draws over the shown environment's links.
struct RobotOverlays {
    bool on = false;
    bool frames = true;       // each link's axes
    bool axes = true;         // joint axes with their limits
    bool com = true;          // each link's and the robot's centre of mass
    bool inertia = false;     // equivalent uniform boxes
    bool translucent = true;  // the robot's meshes while the overlays show
};

// The Measure tool's two ends: a link's origin or centre of mass, a joint's origin, or the ground
// below the first end; it follows the links as they move.
struct MeasureEnd {
    enum class Kind : uint8_t { None, Link, Joint, Ground };
    Kind kind = Kind::None;
    int link = -1;  // by index into the links; a joint end names its child link
};

struct Measure {
    bool com = false;  // a link end takes its centre of mass, else its origin
    MeasureEnd a, b;
};

struct RobotView {
    std::vector<RobotJoint> joints;  // in link order
    std::vector<RobotLink> links;    // in link order
    std::vector<std::string> roots;  // the articulation roots' nodes
    bool floating = false;           // some root is a free base
    JointFrame live;
    std::vector<math::Transform> link_poses;  // the shown environment's, in link order
    bool watch = false;  // the physics thread samples every step
    uint64_t session = 0u;
    uint64_t first = 0u;  // the number of history.front() in the worker's count
    std::deque<JointSample> history;
    int focus = -1;  // the selected link's joint, which the plot follows, by index into `joints`
    int plotted = -1;  // the joint the plot showed last frame; plotting a new one starts the watch
    std::string filter;
    float drive_kp = 100.0f;  // the gain Position drive converts with
    std::vector<JointPose> poses;  // the document's pose library
    float pose_blend = 0.5f;       // the simulated seconds a click on a pose blends over
    GainTuner tuner;
    SysIdView sysid;
    Assembler assembler;
    IkState ik;
    RobotOverlays overlays;
    Measure measure;
    bool radians = false;  // revolute joints show radians, else degrees
    int dragging = -1;     // the joint whose target the panel drags, by index into `joints`
    int hovered = -1;      // the link under the pointer, by index into the links
};

}  // namespace nuka::runtime::app::viewer
