#include <vulkan/vulkan.h>
#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include "lodepng.h" // Used for png encoding.

const int WIDTH = 3200; // Size of rendered Mandelbrot set.
const int HEIGHT = 2400; // Size of rendered Mandelbrot set.
const int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f)                                           \
{                                                                    \
    VkResult res = (f);                                              \
    if (res != VK_SUCCESS)                                           \
    {                                                                \
        printf("Fatal: VkResult is %d in %s at line %d\n", res, __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);                                   \
    }                                                                \
}

uint32_t generateSeed() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

class ComputeApplication {
private:
    struct Pixel {
        float r, g, b, a;
    };

    VkInstance instance;
    VkDebugReportCallbackEXT debugReportCallback;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    uint32_t bufferSize;

    VkBuffer seedBuffer;
    VkDeviceMemory seedBufferMemory;
    uint32_t seedBufferSize = sizeof(uint32_t);

    std::vector<const char*> enabledLayers;
    VkQueue queue;
    uint32_t queueFamilyIndex;

public:
    void run() {
        bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;
        createInstance();
        findPhysicalDevice();
        createDevice();
        createBuffer();
        createSeedBuffer();
        createDescriptorSetLayout();
        createDescriptorSet();
        createComputePipeline();
        createCommandBuffer();
        runCommandBuffer();
        saveRenderedImage();
        cleanup();
    }

    void createSeedBuffer() {
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = seedBufferSize;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &seedBuffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, seedBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, NULL, &seedBufferMemory));
        VK_CHECK_RESULT(vkBindBufferMemory(device, seedBuffer, seedBufferMemory, 0));
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {
        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;

        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool));

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = bufferSize;

        VkDescriptorBufferInfo seedBufferInfo = {};
        seedBufferInfo.buffer = seedBuffer;
        seedBufferInfo.offset = 0;
        seedBufferInfo.range = seedBufferSize;

        VkWriteDescriptorSet descriptorWrites[2] = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &seedBufferInfo;

        vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, NULL);
    }

    void runCommandBuffer() {
        uint32_t newSeed = generateSeed();
        void* data;
        vkMapMemory(device, seedBufferMemory, 0, seedBufferSize, 0, &data);
        memcpy(data, &newSeed, seedBufferSize);
        vkUnmapMemory(device, seedBufferMemory);

        // Other command buffer operations...
    }

    // Other methods omitted for brevity...
};

int main() {
    ComputeApplication app;
    try {
        app.run();
    } catch (const std::runtime_error& e) {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

