#pragma once

#include "../Module.hpp"
#include <bedrocktools/sdk/Types.hpp>
#include <cstdint>
#include <string>

// Xray: highlights ores with colored outlines that can be seen through terrain.
// Scans the blocks around the player a little each frame, caches the ores it
// finds and draws a box around each one after the level has been rendered.
class XrayModule : public Module {
public:
    struct OreType {
        const char* key;        // config toggle name, e.g. "diamond"
        const char* colorKey;   // config color name, e.g. "diamondColor"
        const char* token;      // substring of the block id, e.g. "diamond_ore"
        bool enabled;
        uint32_t color;         // 0xAARRGGBB
    };

    static constexpr int kOreCount = 10;

    XrayModule();
    ~XrayModule() override;

    void onInit() override;
    void onEnable() override;
    void onDisable() override;
    void loadConfig(const nlohmann::json& j) override;
    void saveConfig(nlohmann::json& j) override;

    int radiusHorizontal = 16;
    int radiusVertical = 16;
    std::string material = "name_tag";
    OreType ores[kOreCount];

private:
    void applyPatch();

    bool m_patched = false;
    void* m_patchTarget = nullptr;
};
