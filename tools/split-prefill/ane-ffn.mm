// ane-ffn.mm - see ane-ffn.h
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>

#include "ane-ffn.h"

#include <chrono>
#include <vector>

namespace spt {

struct ane_layer {
    MLModel *      model = nil;
    MLMultiArray * x     = nil;   // (1, S, 1, D) fp16
    MLMultiArray * y     = nil;
    id<MLFeatureProvider> in = nil;
    MLPredictionOptions * opt = nil;
};

struct ane_ffn {
    int il0 = 0, il1 = 0, S = 0, D = 0;
    bool chan_major = false;         // (1, D, 1, S): the ANE layout, transposed here (the fast one); else (1, S, 1, D)
    std::vector<float> tmp;          // [D][S] staging for the transposes
    std::vector<ane_layer> layers;   // indexed by tail il - il0
    ane_ffn_stats st;
    std::string err;
};

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

ane_ffn * ane_ffn_open(const std::string & dir, int il0, int il1, int il_offset, std::string & err) {
    @autoreleasepool {
        auto * a = new ane_ffn();
        a->il0 = il0; a->il1 = il1;
        MLModelConfiguration * cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        for (int il = il0; il < il1; il++) {
            const std::string path = dir + "/ffn_L" + std::to_string(il + il_offset) + ".mlmodelc";
            NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
            NSError * e = nil;
            ane_layer L;
            L.model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&e];
            if (!L.model) {
                err = "ANE FFN: cannot load " + path + ": " + (e ? e.localizedDescription.UTF8String : "?");
                ane_ffn_close(a);
                return nullptr;
            }
            MLFeatureDescription * fd = L.model.modelDescription.inputDescriptionsByName[@"x"];
            NSArray<NSNumber *> * shape = fd.multiArrayConstraint.shape;
            if (!fd || shape.count != 4) { err = "ANE FFN: " + path + " has no (1, S, 1, D) input x"; ane_ffn_close(a); return nullptr; }
            // (1, D, 1, S) channels x tokens (D is the larger dim), or the older token-major (1, S, 1, D)
            const bool cm = shape[1].intValue > shape[3].intValue;
            const int S = cm ? shape[3].intValue : shape[1].intValue, D = cm ? shape[1].intValue : shape[3].intValue;
            if ((a->S && (S != a->S || D != a->D || cm != a->chan_major))) { err = "ANE FFN: models disagree on S/D/layout"; ane_ffn_close(a); return nullptr; }
            a->S = S; a->D = D; a->chan_major = cm;
            a->tmp.resize((size_t) S * D);
            L.x = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&e];
            L.y = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&e];
            if (!L.x || !L.y) { err = "ANE FFN: MLMultiArray alloc failed"; ane_ffn_close(a); return nullptr; }
            L.in  = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x" : L.x } error:&e];
            L.opt = [MLPredictionOptions new];
            L.opt.outputBackings = @{ @"y" : L.y };
            // first prediction compiles/loads the ANE program; do it now, not inside the first prefill
            if (![L.model predictionFromFeatures:L.in options:L.opt error:&e]) {
                err = "ANE FFN: warmup of " + path + " failed: " + (e ? e.localizedDescription.UTF8String : "?");
                ane_ffn_close(a);
                return nullptr;
            }
            a->layers.push_back(L);
        }
        return a;
    }
}

void ane_ffn_close(ane_ffn * a) {
    if (!a) return;
    @autoreleasepool { a->layers.clear(); }
    delete a;
}

// f32 rows [n][D] (row stride D) <-> fp16 rows with row stride rs (elements)
static void f32_to_f16_rows(const float * src, uint16_t * dst, int n, int D, long rs) {
    for (int i = 0; i < n; i++) {
        vImage_Buffer s = { (void *) (src + (size_t) i * D), 1, (vImagePixelCount) D, (size_t) D * 4 };
        vImage_Buffer d = { dst + (size_t) i * rs, 1, (vImagePixelCount) D, (size_t) D * 2 };
        vImageConvert_PlanarFtoPlanar16F(&s, &d, 0);
    }
}
static void f16_to_f32_rows(const uint16_t * src, long rs, float * dst, int n, int D) {
    for (int i = 0; i < n; i++) {
        vImage_Buffer s = { (void *) (src + (size_t) i * rs), 1, (vImagePixelCount) D, (size_t) D * 2 };
        vImage_Buffer d = { dst + (size_t) i * D, 1, (vImagePixelCount) D, (size_t) D * 4 };
        vImageConvert_Planar16FtoPlanarF(&s, &d, 0);
    }
}

bool ane_ffn_run(float * y, const float * x, int32_t n_embd, int32_t n_tok, int32_t il, void * user) {
    auto * a = (ane_ffn *) user;
    const auto t0 = std::chrono::steady_clock::now();
    if (il < a->il0 || il >= a->il1 || n_embd != a->D) {
        a->err = "ANE FFN: layer " + std::to_string(il) + " / n_embd " + std::to_string(n_embd) + " not loaded";
        return false;
    }
    ane_layer & L = a->layers[il - a->il0];
    const int S = a->S, D = a->D;
    @autoreleasepool {
        for (int b0 = 0; b0 < n_tok; b0 += S) {
            const int n = std::min(S, n_tok - b0);
            float * tmp = a->tmp.data();
            const bool cm = a->chan_major;
            [L.x getMutableBytesWithHandler:^(void * p, NSInteger, NSArray<NSNumber *> * strides) {
                const long rs = strides[1].longValue;
                if (cm) {
                    // [n][D] -> [D][n], then each channel row -> fp16, zero-padded to S
                    vDSP_mtrans(x + (size_t) b0 * D, 1, tmp, 1, (vDSP_Length) D, (vDSP_Length) n);
                    for (int c = 0; c < D; c++) {
                        uint16_t * row = (uint16_t *) p + (size_t) c * rs;
                        f32_to_f16_rows(tmp + (size_t) c * n, row, 1, n, 0);
                        if (n < S) memset(row + n, 0, (size_t) (S - n) * 2);
                    }
                } else {
                    f32_to_f16_rows(x + (size_t) b0 * D, (uint16_t *) p, n, D, rs);
                    if (n < S) memset((uint16_t *) p + (size_t) n * rs, 0, (size_t) (S - n) * rs * 2);
                }
            }];
            const auto tp = std::chrono::steady_clock::now();
            NSError * e = nil;
            id<MLFeatureProvider> out = [L.model predictionFromFeatures:L.in options:L.opt error:&e];
            a->st.ms_pred += ms_since(tp);
            if (!out) { a->err = std::string("ANE FFN: prediction failed: ") + (e ? e.localizedDescription.UTF8String : "?"); return false; }
            // CoreML may ignore the backing and return its own array
            MLMultiArray * ya = [out featureValueForName:@"y"].multiArrayValue ?: L.y;
            [ya getBytesWithHandler:^(const void * p, NSInteger) {
                const long rs = ya.strides[1].longValue;
                if (cm) {
                    // each channel row (first n tokens) -> f32 [D][n], then -> [n][D]
                    for (int c = 0; c < D; c++) f16_to_f32_rows((const uint16_t *) p + (size_t) c * rs, 0, tmp + (size_t) c * n, 1, n);
                    vDSP_mtrans(tmp, 1, y + (size_t) b0 * D, 1, (vDSP_Length) n, (vDSP_Length) D);
                } else {
                    f16_to_f32_rows((const uint16_t *) p, rs, y + (size_t) b0 * D, n, D);
                }
            }];
            a->st.blocks++;
        }
    }
    a->st.calls++;
    a->st.ms_total += ms_since(t0);
    return true;
}

ane_ffn_stats ane_ffn_get_stats(const ane_ffn * a) { return a ? a->st : ane_ffn_stats(); }
std::string   ane_ffn_last_error(const ane_ffn * a) { return a ? a->err : std::string(); }

} // namespace spt
