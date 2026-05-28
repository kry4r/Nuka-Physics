// ---------------------------------------------------------------------------
// nuka::tools -- CUDA ABA oracle probe for external reference scripts
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct JointSample {
    std::string name;
    float q = 0.0f;
    float qdot = 0.0f;
    float tau = 0.0f;
};

struct ProbeInput {
    float gravity_z = -9.81f;
    std::vector<JointSample> joints;
};

ProbeInput LoadProbeInput(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open ABA probe input: " + path);
    }

    ProbeInput input;
    std::string tag;
    while (in >> tag) {
        if (tag == "#") {
            std::string rest;
            std::getline(in, rest);
            continue;
        }
        if (tag == "gravity_z") {
            in >> input.gravity_z;
            continue;
        }
        if (tag == "joint") {
            JointSample sample;
            in >> sample.name >> sample.q >> sample.qdot >> sample.tau;
            input.joints.push_back(std::move(sample));
            continue;
        }
        throw std::runtime_error("unknown ABA probe input tag: " + tag);
    }
    return input;
}

std::unordered_map<nuka::scene::BodyId, uint32_t> BuildGlobalLinkMap(
    const std::vector<nuka::runtime::articulation::ArticulationCookedTopology>& topologies) {
    std::unordered_map<nuka::scene::BodyId, uint32_t> result;
    uint32_t offset = 0u;
    for (const auto& topology : topologies) {
        for (uint32_t local = 0u;
             local < static_cast<uint32_t>(topology.link_bodies.size());
             ++local) {
            result[topology.link_bodies[local]] = offset + local;
        }
        offset += static_cast<uint32_t>(topology.link_bodies.size());
    }
    return result;
}

std::unordered_map<std::string, uint32_t> BuildJointLinkMap(
    const nuka::scene::SceneIR& scene,
    const std::unordered_map<nuka::scene::BodyId, uint32_t>& global_link_for_body) {
    std::unordered_map<std::string, uint32_t> result;
    for (const auto& joint : scene.Joints()) {
        const auto link_it = global_link_for_body.find(joint.child_body);
        if (link_it != global_link_for_body.end()) {
            result[joint.name] = link_it->second;
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: nuka_aba_cuda_probe <mjcf.xml> <sample.txt>\n";
            return 2;
        }

        const std::string model_path = argv[1];
        const std::string sample_path = argv[2];
        const ProbeInput input = LoadProbeInput(sample_path);

        const auto scene = nuka::import::LoadMjcf(model_path);
        const auto blob = nuka::scene::CookScene(scene);
        const auto world = nuka::runtime::BuildWorld(blob);
        auto host_state =
            nuka::runtime::articulation::BuildArticulationHostState(
                world.template_view.articulations,
                world.template_view.body_table);

        const auto global_link_for_body =
            BuildGlobalLinkMap(world.template_view.articulations);
        const auto global_link_for_joint =
            BuildJointLinkMap(scene, global_link_for_body);

        for (const auto& sample : input.joints) {
            const auto it = global_link_for_joint.find(sample.name);
            if (it == global_link_for_joint.end()) {
                throw std::runtime_error("joint not found in cooked articulation: " +
                                         sample.name);
            }
            const uint32_t link = it->second;
            if (link >= host_state.q.size()) {
                throw std::runtime_error("joint link out of range for: " + sample.name);
            }
            host_state.q[link] = sample.q;
            host_state.qdot[link] = sample.qdot;
            host_state.tau[link] = sample.tau;
        }

        const auto context = nuka::phi::MakeDefaultDeviceContext();
        auto device_state =
            nuka::runtime::articulation::UploadArticulationState(context, host_state);
        nuka::runtime::articulation::FeatherstoneAba::ComputeAccelerations(
            context,
            device_state.View(),
            input.gravity_z);
        context.stream.Synchronize();

        nuka::runtime::articulation::ArticulationHostState out_state;
        nuka::runtime::articulation::DownloadArticulationState(device_state, &out_state);

        std::cout << std::setprecision(9);
        for (const auto& sample : input.joints) {
            const uint32_t link = global_link_for_joint.at(sample.name);
            std::cout << sample.name << " " << out_state.qddot[link] << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nuka_aba_cuda_probe: " << error.what() << "\n";
        return 1;
    }
}
