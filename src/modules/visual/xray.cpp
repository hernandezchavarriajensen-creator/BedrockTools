#include "xray.hpp"
#include <android/log.h>
#include <bedrocktools/memory/Signatures.hpp>
#include "core/memory/Hooks.hpp"
#include "core/RenderState.hpp"
#include <bedrocktools/sdk/Memory.hpp>
#include <bedrocktools/events/EventBus.hpp>
#include <bedrocktools/sdk/Offsets.hpp>
#include <bedrocktools/sdk/render/Block.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define XRAY_LOG(...) __android_log_print(ANDROID_LOG_INFO, "Xray", __VA_ARGS__)

namespace {

using bedrocktools::sdk::BlockPos;
using bedrocktools::sdk::Vec3;

typedef void (*Tessellator_begin_t)(void* tessellator, void* debugCallback, int primitiveMode, int vertexCount, int noIndices);
typedef void (*Tessellator_color_t)(void* tessellator, float r, float g, float b, float a);
typedef void (*Tessellator_vertex_t)(void* tessellator, float x, float y, float z);
typedef void (*MeshHelpers_renderMeshImmediately_t)(void* screenContext, void* tessellator, void* material, char* pad);
typedef void* (*BlockSource_getBlock_t)(void* region, const BlockPos& pos);

struct HashedString {
    uint64_t mStrHash;
    std::string mStr;
    mutable const HashedString* mLastMatch;

    HashedString() : mStrHash(0), mStr(), mLastMatch(nullptr) {}

    explicit HashedString(const char* str) : mLastMatch(nullptr) {
        mStr = str ? str : "";
        mStrHash = computeHash(mStr);
    }

private:
    static uint64_t computeHash(const std::string& str) {
        if (str.empty()) return 0;
        constexpr uint64_t kOffset = 0xCBF29CE484222325ULL;
        constexpr uint64_t kPrime = 0x100000001B3ULL;
        uint64_t hash = kOffset;
        for (char ch : str)
            hash = static_cast<uint64_t>(static_cast<unsigned char>(ch)) ^ (kPrime * hash);
        return hash;
    }
};

struct MaterialPtr {
    void* sharedPtrData[2]{nullptr, nullptr};

    MaterialPtr() = default;
    MaterialPtr(const MaterialPtr&) = delete;
    MaterialPtr& operator=(const MaterialPtr&) = delete;

    MaterialPtr(MaterialPtr&& other) noexcept
        : sharedPtrData{other.sharedPtrData[0], other.sharedPtrData[1]} {
        other.sharedPtrData[0] = nullptr;
        other.sharedPtrData[1] = nullptr;
    }

    MaterialPtr& operator=(MaterialPtr&& other) noexcept {
        if (this != &other) {
            sharedPtrData[0] = other.sharedPtrData[0];
            sharedPtrData[1] = other.sharedPtrData[1];
            other.sharedPtrData[0] = nullptr;
            other.sharedPtrData[1] = nullptr;
        }
        return *this;
    }

    ~MaterialPtr() {}

    explicit operator bool() const { return sharedPtrData[0] != nullptr; }
};

// Follows an ADRP+ADD (or ADR) pair to find the address a function loads into targetReg.
uintptr_t resolveADRP(uint32_t* insns, size_t count, uint32_t targetReg) {
    for (size_t i = 0; i < count; i++) {
        uint32_t insn = insns[i];
        if ((insn & 0x1F) != targetReg) continue;

        if ((insn & 0x9F000000) == 0x90000000) {
            uintptr_t page = ((uintptr_t)&insns[i] & ~0xFFFULL)
                           + ((int64_t)((uint64_t)((insn >> 3) & 0x1FFFFC | (insn >> 29) & 3) << 43) >> 31);

            for (size_t j = i + 1; j < count; j++) {
                uint32_t add = insns[j];
                if ((add & 0xFF000000) == 0x91000000 &&
                    ((add >> 5) & 0x1F) == targetReg &&
                    (add & 0x1F) == targetReg) {
                    uint32_t imm12 = (add >> 10) & 0xFFF;
                    if (add & 0x400000) imm12 <<= 12;
                    return page + imm12;
                }
                if ((add & 0x1F) == targetReg) break;
            }
        }
        if ((insn & 0x9F000000) == 0x10000000) {
            int64_t imm = (int64_t)((uint64_t)((insn >> 3) & 0x1FFFFC | (insn >> 29)) << 43) >> 43;
            return (uintptr_t)&insns[i] + imm;
        }
    }
    return 0;
}

struct OreHit {
    int x, y, z;
    int ore;
};

constexpr size_t kMaxHits = 6000;   // stored per scan
constexpr size_t kMaxDrawn = 2500;  // boxes drawn per frame (24 vertices each)
constexpr int kMinY = -64;
constexpr int kMaxY = 319;
constexpr int kStaleTickMs = 2000;

XrayModule* g_mod = nullptr;

Tessellator_begin_t s_tessBegin = nullptr;
Tessellator_color_t s_tessColor = nullptr;
Tessellator_vertex_t s_tessVertex = nullptr;
MeshHelpers_renderMeshImmediately_t s_renderMesh = nullptr;
BlockSource_getBlock_t s_getBlock = nullptr;

uintptr_t s_renderMaterialGroup = 0;
MaterialPtr s_material;
std::string s_materialLoaded;
std::string s_materialFailed;

std::mutex s_materialMutex;
std::string s_materialWanted = "name_tag";
std::atomic<uint32_t> s_materialVersion{1};
uint32_t s_materialSeenVersion = 0;

void (*s_renderLevelOrig)(void* self, void* screenContext, void* a3) = nullptr;

// Written by the player tick callback, read by the render hook.
std::atomic<void*> g_player{nullptr};
std::atomic<int64_t> g_lastTickMs{0};
Vec3 g_playerPos = {0.f, 0.f, 0.f};

std::atomic<bool> s_reset{false};

// Render-thread-only state.
std::unordered_map<const void*, int> s_blockCache;  // Block* -> ore index (-1 = not an ore)
std::vector<OreHit> s_hits;
std::vector<OreHit> s_building;
bool s_scanning = false;
int s_scanI = 0;
int s_scanRx = 16;
int s_scanRy = 16;
BlockPos s_scanCenter = {0, 0, 0};
int s_layout = 0;  // 0 = not checked, 1 = block layout verified, -1 = verification failed

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void tickCallback(void* player) {
    if (!g_mod || !g_mod->enabled || !player) return;
    uintptr_t svc = *(uintptr_t*)((uintptr_t)player + bedrocktools::sdk::offsets::Actor::mStateVectorComponent);
    if (svc != 0) g_playerPos = *(Vec3*)svc;
    g_player.store(player);
    g_lastTickMs.store(nowMs());
}

void resetState() {
    s_hits.clear();
    s_building.clear();
    s_blockCache.clear();
    s_scanning = false;
    s_layout = 0;
}

MaterialPtr getMaterial(const char* name) {
    if (!s_renderMaterialGroup) return {};
    HashedString hs(name);
    void** vtable = *reinterpret_cast<void***>(s_renderMaterialGroup);
    if (!vtable || !vtable[bedrocktools::sdk::offsets::VTable::RenderMaterialGroup_getMaterial]) return {};
    using getMat_t = MaterialPtr(*)(void*, const HashedString*);
    return reinterpret_cast<getMat_t>(vtable[bedrocktools::sdk::offsets::VTable::RenderMaterialGroup_getMaterial])(
        (void*)s_renderMaterialGroup, &hs);
}

void ensureMaterial() {
    if (!s_renderMaterialGroup) return;

    const uint32_t version = s_materialVersion.load();
    if (version != s_materialSeenVersion) {
        s_materialSeenVersion = version;
        std::string wanted;
        {
            std::lock_guard<std::mutex> lock(s_materialMutex);
            wanted = s_materialWanted;
        }
        if (wanted.empty()) wanted = "selection_box";
        if (wanted != s_materialLoaded && wanted != s_materialFailed) {
            MaterialPtr m = getMaterial(wanted.c_str());
            if (m) {
                s_material = std::move(m);
                s_materialLoaded = wanted;
                s_materialFailed.clear();
                XRAY_LOG("material '%s' loaded", wanted.c_str());
            } else {
                s_materialFailed = wanted;
                XRAY_LOG("material '%s' not found, keeping the previous one", wanted.c_str());
            }
        }
    }

    if (!s_material && s_materialFailed != "selection_box") {
        MaterialPtr m = getMaterial("selection_box");
        if (m) {
            s_material = std::move(m);
            s_materialLoaded = "selection_box";
        } else {
            s_materialFailed = "selection_box";
        }
    }
}

int matchOre(const std::string& name) {
    if (!g_mod) return -1;
    for (int i = 0; i < XrayModule::kOreCount; ++i) {
        if (name.find(g_mod->ores[i].token) != std::string::npos) return i;
    }
    return -1;
}

const std::string* blockName(const void* block) {
    if (!block) return nullptr;
    const std::string* name = static_cast<const bedrocktools::sdk::Block*>(block)->fullName();
    if (!name || name->size() == 0 || name->size() > 256 || name->data() == nullptr) return nullptr;
    return name;
}

int classifyBlock(const void* block) {
    if (!block) return -1;
    auto it = s_blockCache.find(block);
    if (it != s_blockCache.end()) return it->second;
    int ore = -1;
    if (const std::string* name = blockName(block)) ore = matchOre(*name);
    s_blockCache.emplace(block, ore);
    return ore;
}

// Reads the name of a block next to the player and checks it looks like "namespace:name".
// If the layout of Block/BlockType changed in this Minecraft version the scan stays off
// instead of reading garbage memory for thousands of blocks.
bool verifyBlockLayout(void* region) {
    if (s_layout != 0) return s_layout > 0;
    BlockPos below = {(int)std::floor(g_playerPos.x), (int)std::floor(g_playerPos.y) - 1, (int)std::floor(g_playerPos.z)};
    const void* block = s_getBlock(region, below);
    const std::string* name = blockName(block);
    if (name && name->find(':') != std::string::npos) {
        s_layout = 1;
        XRAY_LOG("block layout OK (block under player: %s)", name->c_str());
        return true;
    }
    s_layout = -1;
    XRAY_LOG("block layout check FAILED, Xray scan disabled for this version");
    return false;
}

void scanStep(void* region) {
    int rx = std::clamp(g_mod->radiusHorizontal, 1, 64);
    int ry = std::clamp(g_mod->radiusVertical, 1, 64);

    if (!s_scanning) {
        s_scanCenter = {(int)std::floor(g_playerPos.x), (int)std::floor(g_playerPos.y), (int)std::floor(g_playerPos.z)};
        s_scanRx = rx;
        s_scanRy = ry;
        s_scanI = -s_scanRx;
        s_building.clear();
        s_scanning = true;
    }

    // One x-layer per frame keeps the cost small and steady.
    const int x = s_scanCenter.x + s_scanI;
    const int yMin = std::max(kMinY, s_scanCenter.y - s_scanRy);
    const int yMax = std::min(kMaxY, s_scanCenter.y + s_scanRy);
    for (int y = yMin; y <= yMax; ++y) {
        for (int z = s_scanCenter.z - s_scanRx; z <= s_scanCenter.z + s_scanRx; ++z) {
            BlockPos bp = {x, y, z};
            const int ore = classifyBlock(s_getBlock(region, bp));
            if (ore < 0 || !g_mod->ores[ore].enabled) continue;
            if (s_building.size() < kMaxHits) s_building.push_back({x, y, z, ore});
        }
    }

    if (++s_scanI > s_scanRx) {
        s_hits.swap(s_building);
        s_building.clear();
        s_scanning = false;
    }
}

void renderLevelHook(void* self, void* screenContext, void* a3) {
    if (s_renderLevelOrig) s_renderLevelOrig(self, screenContext, a3);

    if (s_reset.exchange(false)) resetState();
    if (!g_mod || !g_mod->enabled) return;
    if (!s_tessBegin || !s_tessColor || !s_tessVertex || !s_renderMesh || !s_getBlock) return;
    if (!screenContext || (uintptr_t)screenContext < 0x1000) return;

    // Scan only while the local player is ticking, so no stale world pointers are touched.
    void* player = g_player.load();
    const bool playerFresh = player && (nowMs() - g_lastTickMs.load()) < kStaleTickMs;
    if (playerFresh) {
        uintptr_t dimension = *(uintptr_t*)((uintptr_t)player + bedrocktools::sdk::offsets::Actor::mDimension);
        if (dimension) {
            uintptr_t blockSource = *(uintptr_t*)(dimension + bedrocktools::sdk::offsets::Dimension::mBlockSource);
            if (blockSource && verifyBlockLayout((void*)blockSource)) scanStep((void*)blockSource);
        }
    }
    if (s_hits.empty()) return;

    uintptr_t tessellatorPtr = *(uintptr_t*)((uintptr_t)screenContext + bedrocktools::sdk::offsets::ScreenContext::mTessellator);
    if (!tessellatorPtr || tessellatorPtr < 0x1000) return;
    void* tessellator = (void*)tessellatorPtr;

    uintptr_t lrpPtr = *(uintptr_t*)((uintptr_t)self + bedrocktools::sdk::offsets::LevelRenderer::mLevelRendererPlayer);
    if (!lrpPtr || lrpPtr < 0x1000) return;

    const float camX = *(float*)(lrpPtr + bedrocktools::sdk::offsets::LevelRendererPlayer::mCamPos);
    const float camY = *(float*)(lrpPtr + bedrocktools::sdk::offsets::LevelRendererPlayer::mCamPos + 4);
    const float camZ = *(float*)(lrpPtr + bedrocktools::sdk::offsets::LevelRendererPlayer::mCamPos + 8);

    ensureMaterial();
    void* material = s_material ? (void*)&s_material
                                : (void*)(lrpPtr + bedrocktools::sdk::offsets::LevelRendererPlayer::mSelectionOverlayMaterial);

    bedrocktools::core::ScopedShaderColor shaderColor(screenContext);
    if (!shaderColor) return;

    // Hits that are too far from the player belong to an old scan (teleport, new world).
    const float limitH = (float)std::clamp(g_mod->radiusHorizontal, 1, 64) + 8.0f;
    const float limitV = (float)std::clamp(g_mod->radiusVertical, 1, 64) + 8.0f;

    std::vector<const OreHit*> visible;
    visible.reserve(std::min(s_hits.size(), kMaxDrawn));
    for (const OreHit& hit : s_hits) {
        if (!g_mod->ores[hit.ore].enabled) continue;
        if (std::fabs(hit.x + 0.5f - g_playerPos.x) > limitH) continue;
        if (std::fabs(hit.z + 0.5f - g_playerPos.z) > limitH) continue;
        if (std::fabs(hit.y + 0.5f - g_playerPos.y) > limitV) continue;
        visible.push_back(&hit);
        if (visible.size() >= kMaxDrawn) break;
    }
    if (visible.empty()) return;

    s_tessBegin(tessellator, nullptr, 4, static_cast<int>(visible.size() * 24), 0);

    constexpr float kPad = 0.004f;  // slightly larger than the block to avoid z-fighting
    static const int kEdges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},  // bottom
        {4, 5}, {5, 7}, {7, 6}, {6, 4},  // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7},  // sides
    };

    for (const OreHit* hit : visible) {
        const uint32_t col = g_mod->ores[hit->ore].color;
        s_tessColor(tessellator,
                    ((col >> 16) & 0xFF) / 255.0f,
                    ((col >> 8) & 0xFF) / 255.0f,
                    (col & 0xFF) / 255.0f,
                    ((col >> 24) & 0xFF) / 255.0f);

        const float x0 = hit->x - kPad - camX, x1 = hit->x + 1.0f + kPad - camX;
        const float y0 = hit->y - kPad - camY, y1 = hit->y + 1.0f + kPad - camY;
        const float z0 = hit->z - kPad - camZ, z1 = hit->z + 1.0f + kPad - camZ;
        const Vec3 c[8] = {
            {x0, y0, z0}, {x1, y0, z0}, {x0, y0, z1}, {x1, y0, z1},
            {x0, y1, z0}, {x1, y1, z0}, {x0, y1, z1}, {x1, y1, z1},
        };
        for (const auto& e : kEdges) {
            s_tessVertex(tessellator, c[e[0]].x, c[e[0]].y, c[e[0]].z);
            s_tessVertex(tessellator, c[e[1]].x, c[e[1]].y, c[e[1]].z);
        }
    }

    char pad[0x58];
    memset(pad, 0, sizeof(pad));
    s_renderMesh(screenContext, tessellator, material, pad);
}

}  // namespace

XrayModule::XrayModule()
    : Module("Xray", "Outlines ores through walls. Turn it on and off from the mod menu.") {
    showInMenu = true;

    //          toggle          color key            block id token     on     0xAARRGGBB
    ores[0] = {"diamond",      "diamondColor",      "diamond_ore",     true,  0xFF00FFFF};
    ores[1] = {"emerald",      "emeraldColor",      "emerald_ore",     true,  0xFF00FF40};
    ores[2] = {"gold",         "goldColor",         "gold_ore",        true,  0xFFFFD700};
    ores[3] = {"iron",         "ironColor",         "iron_ore",        true,  0xFFE0B090};
    ores[4] = {"coal",         "coalColor",         "coal_ore",        false, 0xFF909090};
    ores[5] = {"redstone",     "redstoneColor",     "redstone_ore",    true,  0xFFFF2020};
    ores[6] = {"lapis",        "lapisColor",        "lapis_ore",       true,  0xFF3050FF};
    ores[7] = {"copper",       "copperColor",       "copper_ore",      false, 0xFFE07030};
    ores[8] = {"quartz",       "quartzColor",       "quartz_ore",      true,  0xFFFFFFFF};
    ores[9] = {"ancientDebris", "ancientDebrisColor", "ancient_debris", true,  0xFFC0702A};

    g_mod = this;
}

XrayModule::~XrayModule() {
    if (g_mod == this) g_mod = nullptr;
}

void XrayModule::onInit() {
    using bedrocktools::memory::SignatureId;
    using bedrocktools::memory::resolve;

    if (uintptr_t addr = resolve(SignatureId::RenderLevel)) m_patchTarget = (void*)addr;
    if (uintptr_t a = resolve(SignatureId::TessellatorBegin)) s_tessBegin = (Tessellator_begin_t)a;
    if (uintptr_t a = resolve(SignatureId::TessellatorColor)) s_tessColor = (Tessellator_color_t)a;
    if (uintptr_t a = resolve(SignatureId::TessellatorVertex)) s_tessVertex = (Tessellator_vertex_t)a;
    if (uintptr_t a = resolve(SignatureId::MeshHelpersRenderMeshImmediately2)) s_renderMesh = (MeshHelpers_renderMeshImmediately_t)a;
    if (uintptr_t a = resolve(SignatureId::BlockSourceGetBlock)) s_getBlock = (BlockSource_getBlock_t)a;

    if (uintptr_t rmg = resolve(SignatureId::RenderMaterialGroupCommon)) {
        uintptr_t groupAddr = resolveADRP(reinterpret_cast<uint32_t*>(rmg), 2, 0);
        if (groupAddr) s_renderMaterialGroup = groupAddr + bedrocktools::sdk::offsets::MaterialGroup::mRenderMaterialGroupOffset;
    }

    XRAY_LOG("init: renderLevel=%d tessBegin=%d tessColor=%d tessVertex=%d renderMesh=%d getBlock=%d materials=%d",
             m_patchTarget != nullptr, s_tessBegin != nullptr, s_tessColor != nullptr, s_tessVertex != nullptr,
             s_renderMesh != nullptr, s_getBlock != nullptr, s_renderMaterialGroup != 0);

    bedrocktools::events::bus().subscribe<bedrocktools::events::LocalPlayerTickEvent>(
        [](auto& event) { tickCallback(event.player); });
}

void XrayModule::applyPatch() {
    if (m_patched || !m_patchTarget) return;
    bedrocktools::hooks::install(m_patchTarget, (void*)renderLevelHook, (void**)&s_renderLevelOrig);
    m_patched = true;
}

void XrayModule::onEnable() {
    s_reset.store(true);
    applyPatch();
}

void XrayModule::onDisable() {
    s_reset.store(true);
}

void XrayModule::loadConfig(const nlohmann::json& j) {
    Module::loadConfig(j);
    radiusHorizontal = j.value("radiusHorizontal", radiusHorizontal);
    radiusVertical = j.value("radiusVertical", radiusVertical);

    for (auto& ore : ores) {
        ore.enabled = j.value(ore.key, ore.enabled);
        if (j.contains(ore.colorKey)) {
            std::string hex = j[ore.colorKey].get<std::string>();
            if (!hex.empty() && hex[0] == '#') {
                try { ore.color = static_cast<uint32_t>(std::stoul(hex.substr(1), nullptr, 16)); } catch (...) {}
            }
        }
    }

    // The render thread picks up a new material name the next frame (see ensureMaterial).
    std::string wanted = material;
    if (wanted != material) {
        material = wanted;
        {
            std::lock_guard<std::mutex> lock(s_materialMutex);
            s_materialWanted = material;
        }
        s_materialVersion.fetch_add(1);
    }
}

void XrayModule::saveConfig(nlohmann::json& j) {
    Module::saveConfig(j);
    j["radiusHorizontal"] = radiusHorizontal;
    j["radiusVertical"] = radiusVertical;

    for (auto& ore : ores) {
        j[ore.key] = ore.enabled;
        char hex[12];
        snprintf(hex, sizeof(hex), "#%08X", ore.color);
        j[ore.colorKey] = std::string(hex);
    }

    j["material"] = material;
}
