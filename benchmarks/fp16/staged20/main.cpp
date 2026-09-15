// Author: even
// Host harness for the fixed-size FFT benchmarks.
#include <acl/acl.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "aclrtlaunch_fft_step_0.h"
#include "aclrtlaunch_fft_step_1.h"
#include "aclrtlaunch_fft_step_2.h"
#include "aclrtlaunch_fft_step_3.h"
#include "aclrtlaunch_fft_step_4.h"
#include "aclrtlaunch_fft_step_5.h"
#include "aclrtlaunch_fft_step_6.h"
#include "aclrtlaunch_fft_step_7.h"
#include "aclrtlaunch_fft_step_8.h"
#include "aclrtlaunch_fft_step_9.h"
#include "aclrtlaunch_fft_step_10.h"
#include "aclrtlaunch_fft_step_11.h"
#include "aclrtlaunch_fft_step_12.h"
#include "aclrtlaunch_fft_step_13.h"
#include "aclrtlaunch_fft_step_14.h"
#include "aclrtlaunch_fft_step_15.h"
#include "aclrtlaunch_fft_step_16.h"
#include "aclrtlaunch_fft_step_17.h"
#include "aclrtlaunch_fft_step_18.h"
#include "aclrtlaunch_fft_step_19.h"

using Scalar = uint16_t;
constexpr size_t kScalars = 512 * 2;
constexpr size_t kBytes = kScalars * sizeof(Scalar);
constexpr int kWarmup = 1000;
constexpr int kGroups = 11;

void ck(aclError error, const char* operation) {
    if (error) { std::fprintf(stderr, "ACL %d %s\n", error, operation); std::exit(2); }
}

int integer(const char* text, int low, int high) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end || value < low || value > high) {
        std::fprintf(stderr, "invalid integer argument: %s\n", text); std::exit(1);
    }
    return static_cast<int>(value);
}

std::vector<Scalar> read_input(const std::string& path) {
    std::vector<Scalar> value(kScalars);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(value.data()), kBytes) || file.peek() != EOF) {
        std::fprintf(stderr, "expected exactly %zu bytes: %s\n", kBytes, path.c_str()); std::exit(3);
    }
    return value;
}

void save(const std::string& path, const std::vector<Scalar>& value) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(value.data()), kBytes);
    file.close();
    if (!file) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(3); }
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "usage: sage_fft INPUT_DIR OUTPUT [ITERATIONS=10000] [BLOCKS=4] [DEVICE=0]\n");
        return 1;
    }
    const int iters = argc > 3 ? integer(argv[3], 1, 100000) : 10000;
    const int blocks = argc > 4 ? integer(argv[4], 1, 8) : 4;
    const int device = argc > 5 ? integer(argv[5], 0, 1023) : 0;
    if (blocks != 1 && blocks != 4 && blocks != 8) {
        std::fprintf(stderr, "supported block counts are 1, 4, 8\n"); return 1;
    }
    auto xh = read_input(std::string(argv[1]) + "/x.bin");
    auto hh = read_input(std::string(argv[1]) + "/h.bin");
    ck(aclInit(nullptr), "init");
    ck(aclrtSetDevice(device), "device");
    aclrtStream stream;
    ck(aclrtCreateStream(&stream), "stream");
    void *x, *h, *b, *c;
    ck(aclrtMalloc(&x, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    ck(aclrtMalloc(&h, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc h");
    ck(aclrtMalloc(&b, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc b");
    ck(aclrtMalloc(&c, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc c");
    ck(aclrtMemcpy(x, kBytes, xh.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x");
    ck(aclrtMemcpy(h, kBytes, hh.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy h");
    void* result = nullptr;
    auto run = [&]() {
        // The original input remains immutable across all repetitions.
        void* src = x;
        void* dst = b;
        ck(ACLRT_LAUNCH_KERNEL(fft_step_0)(blocks, stream, src, h, dst), "launch 0");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_1)(blocks, stream, src, h, dst), "launch 1");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_2)(blocks, stream, src, h, dst), "launch 2");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_3)(blocks, stream, src, h, dst), "launch 3");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_4)(blocks, stream, src, h, dst), "launch 4");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_5)(blocks, stream, src, h, dst), "launch 5");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_6)(blocks, stream, src, h, dst), "launch 6");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_7)(blocks, stream, src, h, dst), "launch 7");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_8)(blocks, stream, src, h, dst), "launch 8");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_9)(blocks, stream, src, h, dst), "launch 9");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_10)(blocks, stream, src, h, dst), "launch 10");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_11)(blocks, stream, src, h, dst), "launch 11");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_12)(blocks, stream, src, h, dst), "launch 12");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_13)(blocks, stream, src, h, dst), "launch 13");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_14)(blocks, stream, src, h, dst), "launch 14");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_15)(blocks, stream, src, h, dst), "launch 15");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_16)(blocks, stream, src, h, dst), "launch 16");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_17)(blocks, stream, src, h, dst), "launch 17");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_18)(blocks, stream, src, h, dst), "launch 18");
        src = dst; dst = (dst == b ? c : b);
        ck(ACLRT_LAUNCH_KERNEL(fft_step_19)(blocks, stream, src, h, dst), "launch 19");
        src = dst; dst = (dst == b ? c : b);
        result = src;
    };
    std::vector<Scalar> output(kScalars);
    auto download = [&](const std::string& path) {
        ck(aclrtMemcpy(output.data(), kBytes, result, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "output");
        save(path, output);
    };
    run();
    ck(aclrtSynchronizeStream(stream), "check sync");
    download(argv[2]);
    for (int w = 0; w < kWarmup; ++w) run();
    ck(aclrtSynchronizeStream(stream), "warmup sync");
    std::vector<double> samples;
    for (int group = 0; group < kGroups; ++group) {
        auto begin = std::chrono::steady_clock::now();
        for (int j = 0; j < iters; ++j) run();
        ck(aclrtSynchronizeStream(stream), "timing sync");
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count() / iters);
    }
    download(std::string(argv[2]) + ".after");
    std::printf("{\"timing_method\":\"host_batch_submit_sync\",\"unit\":\"us_per_pipeline\","
                "\"precision\":\"fp16\",\"warmup\":%d,\"groups\":%d,\"iterations\":%d,"
                "\"blocks\":%d,\"device_index\":%d,\"launches\":20,\"samples_us\":[",
                kWarmup, kGroups, iters, blocks, device);
    for (size_t j = 0; j < samples.size(); ++j) std::printf("%s%.9f", j ? "," : "", samples[j]);
    std::printf("]}\n");
    ck(aclrtFree(x), "free x"); ck(aclrtFree(h), "free h");
    ck(aclrtFree(b), "free b"); ck(aclrtFree(c), "free c");
    ck(aclrtDestroyStream(stream), "destroy stream");
    ck(aclrtResetDevice(device), "reset device"); ck(aclFinalize(), "finalize");
    return 0;
}
