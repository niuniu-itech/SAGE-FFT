// Native vendor FFTWithSize pipeline using built-in inverse normalization.
// Independent three-operator variant; original four-operator benchmark is unchanged.
// INPUTDIR must contain x.bin/h.bin (complex64), reference.bin and
// reference_fft.bin (complex128). All files use C-order interleaved complex data.
// Build on the target CANN SDK with -lascendcl -lacl_op_compiler -ldl.
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Complex32 { float r, i; };
struct Complex64 { double r, i; };
static_assert(sizeof(Complex32) == 8, "complex64 file layout");
static_assert(sizeof(Complex64) == 16, "complex128 file layout");

void check(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        const char* recent = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(operation) + " returned " + std::to_string(status) +
                                 (recent ? std::string(": ") + recent : ""));
    }
}

std::string quote(const std::string& text) {
    std::ostringstream s;
    s << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') s << '\\' << c;
        else if (c == '\n') s << "\\n";
        else if (c == '\r') s << "\\r";
        else if (c == '\t') s << "\\t";
        else if (c < 32) s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else s << c;
    }
    s << '"';
    return s.str();
}

int parse_int(const char* text, int low, int high) {
    char* end = nullptr;
    long n = std::strtol(text, &end, 10);
    if (end == text || *end || n < low || n > high) throw std::runtime_error("invalid integer argument");
    return static_cast<int>(n);
}

bool exists(const std::string& path) { return std::ifstream(path, std::ios::binary).good(); }

template<class T> std::vector<T> read_exact(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    std::vector<T> value(count);
    const size_t bytes = count * sizeof(T);
    if (!file.read(reinterpret_cast<char*>(value.data()), bytes) || file.peek() != EOF)
        throw std::runtime_error("unexpected input length: " + path);
    for (const auto& c : value)
        if (!std::isfinite(c.r) || !std::isfinite(c.i)) throw std::runtime_error("nonfinite value: " + path);
    return value;
}

void write_output(const std::string& path, const std::vector<Complex32>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(Complex32));
    file.close();
    if (!file) throw std::runtime_error("cannot write: " + path);
}

struct ErrorMetrics {
    bool passed = false;
    bool finite = true;
    double max_abs = 0;
    double relative_l2 = 0;
    std::string json() const {
        std::ostringstream s;
        s << std::setprecision(17) << "{\"passed\":" << (passed ? "true" : "false")
          << ",\"finite\":" << (finite ? "true" : "false")
          << ",\"max_abs\":" << max_abs << ",\"relative_l2\":" << relative_l2
          << ",\"atol\":0.0001,\"rtol\":0.0001}";
        return s.str();
    }
};

ErrorMetrics compare(const std::vector<Complex32>& actual, const std::vector<Complex64>& reference) {
    ErrorMetrics m;
    m.passed = true;
    double error_square = 0, reference_square = 0;
    for (size_t j = 0; j < actual.size(); ++j) {
        if (!std::isfinite(actual[j].r) || !std::isfinite(actual[j].i)) {
            m.passed = m.finite = false;
            continue;
        }
        const double error = std::hypot(double(actual[j].r) - reference[j].r,
                                         double(actual[j].i) - reference[j].i);
        const double magnitude = std::hypot(reference[j].r, reference[j].i);
        m.max_abs = std::max(m.max_abs, error);
        error_square += error * error;
        reference_square += magnitude * magnitude;
        if (error > 1e-4 + 1e-4 * magnitude) m.passed = false;
    }
    // The generated random inputs have nonzero references. A zero-reference
    // failure keeps a finite diagnostic rather than emitting invalid JSON inf.
    m.relative_l2 = reference_square ? std::sqrt(error_square / reference_square) : std::sqrt(error_square);
    return m;
}

struct Tensor {
    void* memory = nullptr;
    aclTensorDesc* desc = nullptr;
    aclDataBuffer* buffer = nullptr;
    size_t bytes = 0;
};

struct Operation {
    const char* name;
    std::vector<const aclTensorDesc*> in_desc;
    std::vector<const aclDataBuffer*> in_buf;
    std::vector<const aclTensorDesc*> out_desc;
    std::vector<aclDataBuffer*> out_buf;
    aclopAttr* attr;

    void compile(aclrtStream stream) {
        check(aclopCompileAndExecute(name, static_cast<int>(in_desc.size()), in_desc.data(), in_buf.data(),
              static_cast<int>(out_desc.size()), out_desc.data(), out_buf.data(), attr,
              ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream), "aclopCompileAndExecute");
    }
    void execute(aclrtStream stream) {
        // Deliberately no fallback to CompileAndExecute. A cache miss fails.
        check(aclopExecute(name, static_cast<int>(in_desc.size()), in_desc.data(), in_buf.data(),
              static_cast<int>(out_desc.size()), out_desc.data(), out_buf.data(), attr, stream), "aclopExecute");
    }
};

std::string symbol_library(const void* symbol) {
    Dl_info info{};
    return dladdr(symbol, &info) && info.dli_fname ? info.dli_fname : "unknown";
}

class Pipeline {
public:
    aclrtStream stream = nullptr;
    std::vector<Tensor> tensors;
    std::vector<aclopAttr*> attrs;
    std::vector<Operation> operations;
    bool initialized = false;
    bool device_set = false;
    int output_index = 4;

    ~Pipeline() {
        if (stream) aclrtSynchronizeStream(stream);
        for (auto& op : attrs) aclopDestroyAttr(op);
        for (auto& t : tensors) {
            if (t.buffer) aclDestroyDataBuffer(t.buffer);
            if (t.desc) aclDestroyTensorDesc(t.desc);
            if (t.memory) aclrtFree(t.memory);
        }
        if (stream) aclrtDestroyStream(stream);
        if (device_set) aclrtResetDevice(0);
        if (initialized) aclFinalize();
    }

    void init() {
        check(aclInit(nullptr), "aclInit"); initialized = true;
        check(aclrtSetDevice(0), "aclrtSetDevice"); device_set = true;
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
        tensors.reserve(5);
    }

    void tensor(const std::vector<int64_t>& shape, const std::vector<Complex32>* host = nullptr) {
        size_t count = 1;
        for (auto n : shape) count *= static_cast<size_t>(n);
        tensors.emplace_back();
        Tensor& t = tensors.back();
        t.bytes = count * sizeof(Complex32);
        check(aclrtMalloc(&t.memory, t.bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        t.desc = aclCreateTensorDesc(ACL_COMPLEX64, static_cast<int>(shape.size()), shape.data(), ACL_FORMAT_ND);
        t.buffer = aclCreateDataBuffer(t.memory, t.bytes);
        if (!t.desc || !t.buffer) throw std::runtime_error("descriptor or buffer construction failed");
        if (host) {
            if (host->size() != count) throw std::runtime_error("host tensor size mismatch");
            check(aclrtMemcpy(t.memory, t.bytes, host->data(), t.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "input H2D");
        }
    }

    aclopAttr* attributes(bool fft, bool inverse = false) {
        aclopAttr* a = aclopCreateAttr();
        if (!a) throw std::runtime_error("aclopCreateAttr failed");
        attrs.push_back(a);
        if (fft) {
            check(aclopSetAttrInt(a, "signal_ndim", 3), "signal_ndim");
            check(aclopSetAttrBool(a, "inverse", inverse), "inverse");
            check(aclopSetAttrBool(a, "real", false), "real");
            check(aclopSetAttrString(a, "norm", "backward"), "norm");
            check(aclopSetAttrBool(a, "onesided", false), "onesided");
            check(aclopSetAttrListInt(a, "signal_sizes", 0, nullptr), "signal_sizes");
        }
        return a;
    }

    void operation(const char* name, const std::vector<int>& inputs, int output, aclopAttr* attr) {
        Operation op{name, {}, {}, {tensors[output].desc}, {tensors[output].buffer}, attr};
        for (int index : inputs) {
            op.in_desc.push_back(tensors[index].desc);
            op.in_buf.push_back(tensors[index].buffer);
        }
        operations.push_back(op);
    }

    void run() { for (auto& op : operations) op.execute(stream); }
    void synchronize() { check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream"); }
    std::vector<Complex32> download(int index) {
        auto& t = tensors[index];
        std::vector<Complex32> host(t.bytes / sizeof(Complex32));
        check(aclrtMemcpy(host.data(), t.bytes, t.memory, t.bytes, ACL_MEMCPY_DEVICE_TO_HOST), "output D2H");
        return host;
    }
};

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: native_fft_builtin_norm INPUTDIR OUTPUTPREFIX N ITERS WARMUPS GROUPS\n";
        std::cerr << "Uses device 0; inverse FFTWithSize applies 1/M internally.\n";
        return 1;
    }
    const std::string input = argv[1], prefix = argv[2];
    std::vector<double> samples;
    ErrorMetrics forward_metrics, before_metrics, after_metrics;
    bool accepted = false, setup_complete = false;
    std::string failure, soc, compiler_library, execute_library;
    int n = 0, iters = 0, warmups = 0, groups = 0;
    try {
        n = parse_int(argv[3], 8, 64);
        if (n != 8 && n != 16 && n != 32 && n != 64) throw std::runtime_error("N must be 8,16,32,64");
        iters = parse_int(argv[4], 1, 1000000);
        warmups = parse_int(argv[5], 0, 1000000);
        groups = parse_int(argv[6], 1, 10000);
        for (const char* suffix : {".before.bin", ".after.bin", ".forward.bin", ".json"})
            if (exists(prefix + suffix)) throw std::runtime_error("existing output preserved: " + prefix + suffix);
        const size_t count = static_cast<size_t>(n) * n * n;
        auto x = read_exact<Complex32>(input + "/x.bin", count);
        auto h = read_exact<Complex32>(input + "/h.bin", count);
        auto reference = read_exact<Complex64>(input + "/reference.bin", count);
        auto reference_fft = read_exact<Complex64>(input + "/reference_fft.bin", count);
        Pipeline p;
        p.init();
        const char* soc_name = aclrtGetSocName();
        soc = soc_name ? soc_name : "unknown";
        const std::vector<int64_t> shape{n, n, n};
        p.tensor(shape, &x);     // 0 immutable input
        p.tensor(shape, &h);     // 1 immutable weights
        p.tensor(shape);        // 2 forward spectrum
        p.tensor(shape);        // 3 weighted spectrum
        p.tensor(shape);        // 4 normalized inverse output
        p.operation("FFTWithSize", {0}, 2, p.attributes(true, false));
        p.operation("Mul", {2, 1}, 3, p.attributes(false));
        p.operation("FFTWithSize", {3}, 4, p.attributes(true, true));
        compiler_library = symbol_library(reinterpret_cast<const void*>(&aclopCompileAndExecute));
        execute_library = symbol_library(reinterpret_cast<const void*>(&aclopExecute));
        // Each exact descriptor/attribute signature is compiled outside timing.
        for (auto& op : p.operations) op.compile(p.stream);
        p.synchronize();
        // Test pure cached execution before any accepted measurement.
        p.run();
        p.synchronize();
        auto forward = p.download(2), before = p.download(4);
        forward_metrics = compare(forward, reference_fft);
        before_metrics = compare(before, reference);
        write_output(prefix + ".forward.bin", forward);
        write_output(prefix + ".before.bin", before);
        if (!forward_metrics.passed || !before_metrics.passed)
            throw std::runtime_error("native forward or pipeline correctness failed before timing");
        setup_complete = true;
        for (int i = 0; i < warmups; ++i) p.run();
        p.synchronize();
        for (int group = 0; group < groups; ++group) {
            p.synchronize();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) p.run();
            p.synchronize();
            const auto end = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
            if (!std::isfinite(us) || us <= 0) throw std::runtime_error("invalid timing sample");
            samples.push_back(us);
            std::cerr << "GROUP " << group << " " << std::setprecision(10) << us << " us_per_pipeline\n";
        }
        auto after = p.download(4);
        write_output(prefix + ".after.bin", after);
        after_metrics = compare(after, reference);
        if (!after_metrics.passed) throw std::runtime_error("pipeline correctness failed after timing");
        accepted = true;
    } catch (const std::exception& e) {
        failure = e.what();
        std::cerr << failure << '\n';
    }
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"implementation\":\"native ACL FFTWithSize pipeline with built-in inverse normalization\",\"soc\":" << quote(soc)
         << ",\"shape\":[" << n << ',' << n << ',' << n << "],\"batch\":1,\"precision\":\"complex64\""
         << ",\"warmup\":" << warmups << ",\"groups\":" << groups << ",\"iterations\":" << iters
         << ",\"device_index\":0,\"pipeline_operator_calls\":3,\"timing_method\":\"host_batch_submit_sync\""
         << ",\"unit\":\"us_per_complete_pipeline\",\"clock\":\"std::chrono::steady_clock\""
         << ",\"resident_inputs\":true,\"immutable_input\":true,\"framework_dispatch\":false"
         << ",\"compile_api\":\"aclopCompileAndExecute\",\"timed_api\":\"aclopExecute\""
         << ",\"compile_calls_inside_timing\":0,\"cache_failure_fallback\":false"
         << ",\"compiler_library\":" << quote(compiler_library) << ",\"execute_library\":" << quote(execute_library)
         << ",\"fft_backend\":\"vendor custom AI CPU FFTWithSize\",\"mul_backend\":\"installed CANN dispatch; inspect runtime trace\""
         << ",\"normalization\":\"inverse FFTWithSize norm=backward applies 1/M once\",\"scale_binding\":\"built into inverse; no scale tensor or separate normalize operator\""
         << ",\"custom_opp_path\":" << quote(std::getenv("ASCEND_CUSTOM_OPP_PATH") ? std::getenv("ASCEND_CUSTOM_OPP_PATH") : "")
         << ",\"correctness\":{\"forward\":" << forward_metrics.json()
         << ",\"pipeline_before\":" << before_metrics.json() << ",\"pipeline_after\":" << after_metrics.json() << '}'
         << ",\"cached_execution_validated\":" << (setup_complete ? "true" : "false")
         << ",\"accepted_timing\":" << (accepted ? "true" : "false") << ",\"samples_us\":[";
    for (size_t i = 0; i < samples.size(); ++i) json << (i ? "," : "") << samples[i];
    json << ']';
    if (accepted) {
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const size_t middle = sorted.size() / 2;
        const double median = sorted.size() % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
        json << ",\"median_us\":" << median;
    }
    if (!failure.empty()) json << ",\"failure\":" << quote(failure);
    json << "}\n";
    // Refuse to overwrite a previous JSON, including the failure path.
    if (!exists(prefix + ".json")) {
        std::ofstream file(prefix + ".json");
        file << json.str(); file.close();
        if (!file) { std::cerr << "cannot write result JSON\n"; return 3; }
    }
    std::cout << json.str();
    return accepted ? 0 : 2;
}
